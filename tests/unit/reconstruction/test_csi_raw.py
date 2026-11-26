"""CSI provenance, multi-channel ordering and candidate diagnostics contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.noise import (
    CANDIDATE_REASONS,
    inspect_csi_noise_candidates,
)
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.reconstruction.csi_raw import (
    audit_csi_raw_input,
    csi_raw_fingerprint,
    sort_csi_encoding_grid,
    strip_csi_storage_padding,
)
from hypermri.reconstruction.strategies.bruker.csi import (
    BrukerCsiRecoStrategy,
    _bruker_storage_record_samples,
)


def describe_from_parameters(handler) -> CSIAcquisition:
    """Describe a synthetic handler from its current encoding parameters.

    The positions come from the production Bruker encoding-order parser, so
    tests that edit ``handler.method`` still exercise its validation. The
    fields that sorting does not read carry neutral invented values.
    """
    order, read, phase = BrukerCsiRecoStrategy().get_phase_encoding_order(handler)
    return CSIAcquisition(
        encoding_order=order,
        matrix=tuple(handler.matrix[:2]),
        fov_mm=None,
        fov_offset_mm=(0.0, 0.0, 0.0),
        read_positions=read,
        phase_positions=phase,
        n_points=handler.n_acquisition,
        dwell_time_s=None,
        reference_frequency_mhz=None,
        center_ppm=0.0,
    )


def handler_fixture(
    *,
    shape=(4, 4),
    channels=2,
    repetitions=1,
    slices=1,
    samples=64,
    method="<User:lucaCSI4>",
    centric=False,
):
    read, phase = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    read, phase = read.ravel(), phase.ravel()
    if centric:
        order = np.argsort(
            (read - shape[0] // 2) ** 2 + (phase - shape[1] // 2) ** 2, kind="stable"
        )
        read, phase = read[order], phase[order]
    count = shape[0] * shape[1]
    rng = np.random.default_rng(60904)
    payload = rng.standard_normal(
        (repetitions, slices, count, channels, samples)
    ) + 1j * rng.standard_normal((repetitions, slices, count, channels, samples))
    metadata = {
        "Method": method,
        "PVM_Matrix": np.array([*shape, slices]),
        "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
        "PVM_EncSteps0": np.arange(shape[0]) - shape[0] // 2,
        "PVM_EncSteps1": np.arange(shape[1]) - shape[1] // 2,
        "CentricEncOrder_OnOff": "On" if centric else "Off",
        "SpiralEncOrder_OnOff": "Off",
        "CentricEncOrderMatrixx": read,
        "CentricEncOrderMatrixy": phase,
    }
    handler = SimpleNamespace(
        fid=payload.reshape(-1),
        rawdatajob0=np.array([], dtype=complex),
        n_repetitions=repetitions,
        n_slices=slices,
        n_receivers=channels,
        n_acquisition=samples,
        matrix=np.array([*shape, slices]),
        method=metadata,
        cut_off=0,
        sampling_time_axis_s=np.arange(samples) * 0.001,
        read_orientation="L_R",
        rps_grid_mm=np.zeros((3, *shape, slices)),
        patient_pos="Head_Prone",
        slice_orientation="axial",
        acquisition_matrix=np.array([*shape, slices]),
        slice_thickness_mm=1.0,
        git_commit_hash=None,
        spatial_dimension="2D",
        resolution_mm=np.ones(3),
        rps_extent_mm=np.array([*shape, slices], dtype=float),
        rps_axes_mm=[np.arange(s) for s in (*shape, slices)],
        slice_order=np.arange(slices),
        is_spectroscopic=True,
        type=method,
        expnum=1,
        read_offset_mm=0,
        phase_offset_mm=0,
        slice_offset_mm=0,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    handler.method_parameter = lambda name: handler.method[name]
    handler.to_csi_acquisition = lambda: describe_from_parameters(handler)
    handler.acqp = {"GO_raw_data_format": "GO_32BIT_SGN_INT", "BYTORDA": "little"}
    return handler, payload, read, phase


def inspect(raw, **kwargs):
    return inspect_csi_noise_candidates(
        raw, diagnostic_samples=slice(0, 32), estimation_samples=slice(32, 64), **kwargs
    )


@pytest.mark.parametrize("centric", [False, True])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_sort_tracks_all_channels_repetitions_and_slices(centric, channels):
    handler, payload, read, phase = handler_fixture(
        shape=(3, 2), channels=channels, repetitions=2, slices=2, centric=centric
    )
    before = handler.fid.copy()
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    for i, (r, p) in enumerate(zip(read, phase, strict=True)):
        actual = raw.isel(read_idx=int(r), phase_idx=int(p)).transpose(
            "repetition_idx", "slice_idx", "channel_idx", "acquisition_idx"
        )
        np.testing.assert_array_equal(actual.values, payload[:, :, i])
    np.testing.assert_array_equal(raw.csi_acquisition_count, np.ones((3, 2), dtype=int))
    assert raw.csi_acquired.all()
    assert raw.attrs["domain"] == "kspace"
    assert raw.attrs["csi_raw_fingerprint"] == csi_raw_fingerprint(raw)
    np.testing.assert_array_equal(handler.fid, before)


def test_duplicates_and_missing_bins_follow_plan_not_signal():
    handler, payload, read, phase = handler_fixture(centric=True)
    read[1], phase[1] = read[0], phase[0]
    handler.method["CentricEncOrderMatrixx"] = read
    handler.method["CentricEncOrderMatrixy"] = phase
    payload[:, :, 4] = 0
    handler.fid = payload.reshape(-1)
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    assert int((raw.csi_acquisition_count == 0).sum()) == 1
    assert int((raw.csi_acquisition_count == 2).sum()) == 1
    np.testing.assert_allclose(
        raw.isel(read_idx=int(read[0]), phase_idx=int(phase[0])).transpose(
            "repetition_idx", "slice_idx", "channel_idx", "acquisition_idx"
        ),
        payload[:, :, :2].mean(axis=2),
    )
    assert np.isnan(raw.where(~raw.csi_acquired, drop=False)).all()
    assert raw.csi_acquired.isel(read_idx=int(read[4]), phase_idx=int(phase[4])).item()
    assert (raw.isel(read_idx=int(read[4]), phase_idx=int(phase[4])) == 0).all()


@pytest.mark.parametrize("bad", [[-1, 0], [0, 2], [0, 0.5], [0, np.nan], [0], [[0, 1]]])
def test_invalid_phase_plan_is_rejected(bad):
    with pytest.raises(ValueError, match="one in-grid integer"):
        sort_csi_encoding_grid(
            np.ones((4, 1, 2, 1, 1), dtype=complex), bad, [0, 0], (2, 1)
        )


@pytest.mark.parametrize("method", ["<Bruker:CSI>", "<User:csi_fix>"])
def test_scanner_grid_is_spatial_even_when_array_is_named_kspace(method):
    handler, *_ = handler_fixture(method=method)
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    raw.name = "kspace_xarray"
    assert raw.attrs["domain"] == raw.attrs["csi_grid_domain"] == "spatial"
    report = inspect(raw)
    assert report.attrs["noise_reason"] == "spatial_input"
    assert not report.candidate_eligible.any()
    assert np.isnan(report.noise_sigma_quadrature).all()


def test_reconstruction_preserves_raw_precision_and_drops_encoding_masks():
    handler, *_ = handler_fixture()
    original = handler.fid.copy()
    strategy = BrukerCsiRecoStrategy()
    raw = strategy.sort_kspace(handler, data_type="fid")
    reco = strategy.reconstruct(
        handler, data_type="fid", data_precision="float16", append_parameters=False
    )
    np.testing.assert_array_equal(handler.fid, original)
    assert handler.fid.dtype == np.complex128
    assert not hasattr(handler, "kspace_xarray")
    assert raw.dtype == np.complex128
    assert reco.dtype == np.complex64
    assert reco.attrs["domain"] == "spatial"
    assert "csi_acquired" not in reco.coords
    assert "csi_acquisition_count" not in reco.coords
    assert "csi_raw_fingerprint" not in reco.attrs
    assert reco.attrs["csi_source_raw_fingerprint"] == raw.attrs["csi_raw_fingerprint"]
    trace = json.loads(reco.attrs["csi_reconstruction_trace"])
    assert trace["spatial_ifft_sizes"] == {"read_idx": 4, "phase_idx": 4}
    assert trace["channel_combination"] == "none"
    assert inspect(reco).attrs["noise_reason"] == "spatial_input"


def test_actual_source_records_fallback_and_job_is_not_auto_verified():
    handler, *_ = handler_fixture()
    sorter = BrukerCsiRecoStrategy()
    fallback = sorter.sort_kspace(handler, data_type="job")
    assert json.loads(fallback.attrs["csi_raw_trace"])["source"] == "fid"
    assert json.loads(fallback.attrs["csi_raw_trace"])["requested_source"] == "job"
    handler.rawdatajob0 = handler.fid.copy()
    job = sorter.sort_kspace(handler, data_type="job")
    assert json.loads(job.attrs["csi_raw_trace"])["source"] == "job"
    assert job.attrs["domain"] == job.attrs["csi_grid_domain"] == "kspace"
    assert inspect(job).attrs["noise_reason"] == "unverified_kspace_domain"
    supplied = sorter.sort_kspace(handler, data=handler.fid)
    assert json.loads(supplied.attrs["csi_raw_trace"])["source"] == "input_data"
    assert supplied.attrs["domain"] == "unknown"
    assert not inspect(supplied).attrs["raw_input_verified"]


def test_noise_and_signal_contamination_are_descriptive_not_probabilities():
    handler, payload, *_ = handler_fixture()
    time = np.arange(64) * 0.001
    payload[0, 0, 0] += 30 * np.exp(2j * np.pi * 125 * time - time / 0.04)
    # An acquired, truly zero input is distinct from an unacquired grid bin.
    payload[0, 0, 3] = 0
    # A second contaminated channel has exactly proportional samples.
    payload[0, 0, 0, 1] = (1 + 2j) * payload[0, 0, 0, 0]
    handler.fid = payload.reshape(-1)
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    before = raw.copy(deep=True)
    report = inspect(raw)
    point = report.isel(read_idx=0, phase_idx=0, repetition_idx=0, slice_idx=0)
    signal = payload[0, 0, 0, 0, :32]
    assert point.mean_energy.isel(channel_idx=0).item() == pytest.approx(
        np.mean(np.abs(signal) ** 2)
    )
    assert point.spectral_concentration.isel(channel_idx=0).item() > 0.8
    assert point.lag1_correlation_magnitude.isel(channel_idx=0).item() > 0.95
    assert point.late_to_early_energy_ratio.isel(channel_idx=0).item() < 0.6
    assert point.channel_correlation_magnitude.isel(
        channel_idx=0, channel_partner=1
    ).item() == pytest.approx(1)
    assert report.candidate_eligible.any()
    assert report.acquired.isel(read_idx=3, phase_idx=0).all()
    assert (
        report.candidate_rejection_reason.isel(read_idx=3, phase_idx=0).values
        & CANDIDATE_REASONS["zero_energy"]
    ).all()
    assert report.attrs["noise_status"] == "insufficient"
    assert report.attrs["noise_reason"] == "calibration_required"
    assert not report.selected_mask.any()
    assert np.isnan(report.p_noise_dominated).all()
    assert np.isnan(report.noise_sigma_quadrature).all()
    xr.testing.assert_identical(raw, before)


@pytest.mark.parametrize(
    "operation", ["filter", "pad", "mask", "times", "domain", "combine"]
)
def test_stale_raw_provenance_rejects_processed_arrays(operation):
    handler, *_ = handler_fixture()
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid").copy(deep=True)
    if operation == "filter":
        raw.data *= np.linspace(1, 0.1, 64)[:, None, None, None, None, None]
    elif operation == "pad":
        raw = raw.pad(read_idx=(1, 1), constant_values=0)
    elif operation == "mask":
        raw.coords["csi_acquisition_count"].values[0, 0] = 3
    elif operation == "times":
        raw.coords["acquisition_time_s"].values[1] += 0.1
    elif operation == "domain":
        raw.attrs["domain"] = "spectral"
    else:
        raw = raw.sum("channel_idx", keep_attrs=True).expand_dims(channel_idx=[0])
    report = inspect(raw)
    assert not report.attrs["raw_input_verified"]
    assert not report.candidate_eligible.any()
    assert not report.selected_mask.any()
    assert np.isnan(report.noise_sigma_quadrature).all()


def test_diagnostic_metrics_do_not_use_reserved_estimation_values():
    handler, payload, *_ = handler_fixture()
    first = inspect(BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid"))
    payload[..., 32:] *= 100
    handler.fid = payload.reshape(-1)
    second = inspect(BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid"))
    for name in (
        "mean_energy",
        "spectral_concentration",
        "lag1_correlation_magnitude",
        "candidate_eligible",
        "channel_correlation_magnitude",
    ):
        xr.testing.assert_identical(first[name], second[name])
    assert first.attrs["raw_fingerprint"] != second.attrs["raw_fingerprint"]
    assert first.attrs["split_independence"] == "unverified"


def test_axis_transpose_and_dask_do_not_change_candidate_results():
    pytest.importorskip("dask.array")
    handler, *_ = handler_fixture()
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    reference = inspect(raw)
    transposed = raw.transpose(*reversed(raw.dims))
    assert csi_raw_fingerprint(transposed) == csi_raw_fingerprint(raw)
    result = inspect(transposed.chunk({"acquisition_idx": 8, "read_idx": 2}))
    for name in reference.data_vars:
        xr.testing.assert_identical(
            reference[name], result[name].transpose(*reference[name].dims)
        )


@pytest.mark.parametrize(
    "diagnostic,estimation",
    [
        (slice(0, 32), slice(16, 64)),
        (slice(0, 32, 2), slice(32, 64)),
        (slice(-1, 8), slice(32, 64)),
        (slice(0, 3), slice(32, 64)),
        (slice(0, 32), slice(32, 70)),
    ],
)
def test_explicit_disjoint_blocks_are_required(diagnostic, estimation):
    handler, *_ = handler_fixture()
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    with pytest.raises(ValueError):
        inspect_csi_noise_candidates(
            raw, diagnostic_samples=diagnostic, estimation_samples=estimation
        )


def test_outer_candidates_are_symmetric_and_independent_of_energy():
    handler, *_ = handler_fixture(shape=(5, 3))
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    report = inspect(raw)
    outer = (
        report.outer_candidate.isel(channel_idx=0, slice_idx=0, repetition_idx=0)
        .transpose("read_idx", "phase_idx")
        .values
    )
    np.testing.assert_array_equal(outer, outer[::-1])
    np.testing.assert_array_equal(outer, outer[:, ::-1])
    assert not outer[2, 1]


def test_partial_acquisition_preserves_full_grid_origin_and_missing_mask():
    handler, payload, *_ = handler_fixture(centric=True)
    # Signed positions omit the left-most full-grid column. Origin must not move
    # to the observed minimum (-1 instead of the true full-grid edge -2).
    read = np.array([-1, 0, 1, -1, 0, 1])
    phase = np.array([-1, -1, -1, 0, 0, 0])
    handler.method["CentricEncOrderMatrixx"] = read
    handler.method["CentricEncOrderMatrixy"] = phase
    handler.fid = payload[:, :, :6].reshape(-1)
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    assert int(raw.csi_acquired.sum()) == 6
    assert not raw.csi_acquired.isel(read_idx=0).any()
    assert np.isnan(raw.isel(read_idx=0)).all()
    for i, (r, p) in enumerate(zip(read + 2, phase + 2, strict=True)):
        np.testing.assert_array_equal(
            raw.isel(read_idx=int(r), phase_idx=int(p)).transpose(
                "repetition_idx", "slice_idx", "channel_idx", "acquisition_idx"
            ),
            payload[:, :, i],
        )
    report = inspect(raw)
    assert report.attrs["raw_input_verified"]
    assert not report.candidate_eligible.isel(read_idx=0).any()


def test_invalid_supplied_payload_cannot_fall_back_to_other_data():
    handler, *_ = handler_fixture()
    with pytest.raises(ValueError, match="no silent fallback"):
        BrukerCsiRecoStrategy().sort_kspace(handler, data=handler.fid[:-1])


@pytest.mark.parametrize("missing_fid", [np.array([], dtype=complex), None])
def test_scan_without_raw_data_names_the_missing_files(missing_fid):
    """Regression test: a scan without raw data is rejected by cause.

    A CSI scan whose ``fid`` and ``rawdata.job0`` hold no samples (a scan
    that was set up but not acquired) used to fail with ``No CSI payload
    matches ...``, which reads like a wrong acquisition plan.
    """
    handler, *_ = handler_fixture()
    handler.fid = missing_fid
    handler.load_data = lambda source: None

    with pytest.raises(ValueError, match="has no CSI raw data"):
        BrukerCsiRecoStrategy().reconstruct(handler, append_parameters=False)
    with pytest.raises(ValueError, match="has no CSI raw data"):
        BrukerCsiRecoStrategy().sort_kspace(handler)


def test_payload_of_the_wrong_size_still_reports_the_acquisition_plan():
    handler, *_ = handler_fixture()
    handler.fid = handler.fid[:-1]
    with pytest.raises(ValueError, match="No CSI payload matches"):
        BrukerCsiRecoStrategy().sort_kspace(handler)


def test_unreviewed_spiral_fails_explicitly():
    handler, _, read, phase = handler_fixture()
    handler.method["SpiralEncOrder_OnOff"] = "On"
    handler.method["SpiralEncOrderMatrixx"] = read
    handler.method["SpiralEncOrderMatrixy"] = phase
    with pytest.raises(NotImplementedError, match="verified acquisition-grid sorter"):
        BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")


@pytest.mark.parametrize("method", ["<Bruker:CSI>", "<User:lucaCSI4>"])
def test_different_order_on_the_two_axes_names_the_scanner_parameter(method):
    """Regression: sorting ended in an ``UnboundLocalError``."""
    handler, *_ = handler_fixture(method=method)
    handler.method["PVM_EncOrder"] = ["LINEAR_ENC", "CENTRIC_ENC"]
    with pytest.raises(ValueError, match="PVM_EncOrder='LINEAR_ENC CENTRIC_ENC'"):
        BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")


@pytest.mark.parametrize("shape", [(4, 4), (3, 2), (1, 1)])
def test_unknown_csi_method_is_named_before_any_data_are_sorted(shape):
    """Regression: the error named the payload size or the pairing instead."""
    handler, *_ = handler_fixture(method="<User:otherCSI>", shape=shape)
    with pytest.raises(ValueError, match="Method='<User:otherCSI>'"):
        BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")


@pytest.mark.parametrize("orientation", ["A_P", "L_R"])
def test_stock_centric_positions_are_not_cartesian_expanded_twice(orientation):
    handler, payload, *_ = handler_fixture(method="<Bruker:CSI>", shape=(3, 2))
    handler.method["PVM_EncOrder"] = ["CENTRIC_ENC", "CENTRIC_ENC"]
    handler.method["PVM_EncSteps0"] = np.array([0, -1, 1])
    handler.method["PVM_EncSteps1"] = np.array([0, -1])
    handler.read_orientation = orientation
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    positions = [(r, p) for p in (1, 0) for r in (1, 0, 2)]
    for i, (r, p) in enumerate(positions):
        np.testing.assert_array_equal(
            raw.isel(read_idx=r, phase_idx=p).transpose(
                "repetition_idx", "slice_idx", "channel_idx", "acquisition_idx"
            ),
            payload[:, :, i],
        )
    assert raw.attrs["domain"] == "spatial"


@pytest.mark.parametrize(
    "value,flag",
    [
        (np.nan, "nonfinite_samples"),
        (np.inf, "nonfinite_samples"),
        (1e300, "nonfinite_diagnostics"),
    ],
)
def test_invalid_or_overflowed_candidate_never_becomes_eligible(value, flag):
    handler, payload, *_ = handler_fixture()
    payload[0, 0, 0, 0, 3] = value
    handler.fid = payload.reshape(-1)
    report = inspect(BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid"))
    point = report.isel(
        read_idx=0, phase_idx=0, channel_idx=0, repetition_idx=0, slice_idx=0
    )
    assert not point.candidate_eligible.item()
    assert point.candidate_rejection_reason.item() & CANDIDATE_REASONS[flag]
    assert np.isnan(point.p_noise_dominated.item())


def test_missing_provenance_and_zero_only_reference_remain_insufficient():
    handler, payload, *_ = handler_fixture()
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    raw.attrs = {"domain": "kspace", "signal_domain": "temporal"}
    assert inspect(raw).attrs["noise_reason"] == "missing_raw_provenance"
    payload[:] = 0
    handler.fid = payload.reshape(-1)
    report = inspect(BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid"))
    assert report.attrs["noise_reason"] == "no_eligible_candidates"
    assert not report.selected_mask.any()


def test_declared_storage_padding_is_removed_without_using_sample_values():
    handler, payload, *_ = handler_fixture(channels=1, samples=64)
    # One 64-complex-sample FID = 512 bytes, stored in a 1024-byte int32 block.
    records = payload.reshape(-1, 64)
    padding = np.full_like(records, 321 + 987j)  # never select by zero magnitude
    handler.fid = np.concatenate([records, padding], axis=-1).reshape(-1)
    before = handler.fid.copy()
    handler.acqp = {
        "GO_block_size": "Standard_KBlock_Format",
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "ACQ_size": np.array([128, 4, 4]),
    }
    raw = BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")
    for p in range(4):
        for r in range(4):
            np.testing.assert_array_equal(
                raw.isel(read_idx=r, phase_idx=p).values.ravel(),
                payload[0, 0, p * 4 + r, 0],
            )
    assert (
        json.loads(raw.attrs["csi_raw_trace"])[
            "storage_padding_complex_samples_per_record"
        ]
        == 64
    )
    np.testing.assert_array_equal(handler.fid, before)
    handler.acqp["GO_raw_data_format"] = "GO_32BIT_FLOAT"
    with pytest.raises(ValueError, match="No CSI payload matches"):
        BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid")


def test_storage_padding_follows_the_declared_record_size_only():
    # Invented layout: 2 records of 2 channels x 3 samples, each padded to 8.
    records = np.arange(12, dtype=complex).reshape(2, 6) + 1j
    padded = np.concatenate([records, np.full((2, 2), 5 + 5j)], axis=1).ravel()
    before = padded.copy()

    values, removed = strip_csi_storage_padding(
        padded, samples=3, channels=2, records=2, record_samples=8
    )

    np.testing.assert_array_equal(values, records.ravel())
    assert removed == 2
    np.testing.assert_array_equal(padded, before)
    for record_samples in (None, 9):
        unchanged, removed = strip_csi_storage_padding(
            padded, samples=3, channels=2, records=2, record_samples=record_samples
        )
        assert unchanged is not None and unchanged.size == padded.size
        assert removed == 0
    with pytest.raises(ValueError, match="record_samples=5"):
        strip_csi_storage_padding(
            padded, samples=3, channels=2, records=2, record_samples=5
        )


@pytest.mark.parametrize(
    ("changes", "channels", "expected"),
    [
        ({}, 1, 128),
        ({}, 2, None),
        ({"ACQ_size": np.array([100, 4, 4])}, 1, None),
        ({"GO_block_size": "continuous"}, 1, None),
        ({"BYTORDA": "big"}, 1, None),
    ],
)
def test_bruker_declares_padding_only_for_the_confirmed_layout(
    changes, channels, expected
):
    acqp = {
        "GO_block_size": "Standard_KBlock_Format",
        "GO_raw_data_format": "GO_32BIT_SGN_INT",
        "BYTORDA": "little",
        "ACQ_size": np.array([128, 4, 4]),
        **changes,
    }
    # 64 complex samples x 8 bytes = 512 bytes, stored in one 1024-byte block.
    assert (
        _bruker_storage_record_samples(acqp, samples=64, channels=channels) == expected
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"GO_raw_data_format": "GO_32BIT_FLOAT", "BYTORDA": "little"},
        {"GO_raw_data_format": "GO_32BIT_SGN_INT", "BYTORDA": "big"},
    ],
)
def test_matching_payload_size_does_not_validate_unknown_storage_codec(metadata):
    handler, *_ = handler_fixture()
    handler.acqp = metadata
    report = inspect(BrukerCsiRecoStrategy().sort_kspace(handler, data_type="fid"))
    assert report.attrs["noise_reason"] == "unverified_storage_codec"
    assert not report.candidate_eligible.any()


def test_geoff_csi_is_fourier_encoded_and_gets_phase_shift_and_ifft():
    """geoffCSI raw data are k-space; they share the PVM plan but not the scanner grid."""
    handler, *_ = handler_fixture(method="<User:geoffCSI>")
    strategy = BrukerCsiRecoStrategy()
    raw = strategy.sort_kspace(handler, data_type="fid")
    assert raw.attrs["domain"] == raw.attrs["csi_grid_domain"] == "kspace"
    reco = strategy.reconstruct(handler, data_type="fid", append_parameters=False)
    assert reco.attrs["domain"] == "spatial"
    assert reco.attrs["csi_spatial_reconstruction"] == "ifft"
    trace = json.loads(reco.attrs["csi_reconstruction_trace"])
    assert trace["fov_phase_shift"] == "applied"
    assert trace["spatial_ifft_sizes"] == {"read_idx": 4, "phase_idx": 4}


@pytest.mark.parametrize(("rule", "reason_is_summed"), [("mean", False), ("sum", True)])
def test_density_weighted_sum_is_refused_as_kspace_noise_input(rule, reason_is_summed):
    handler, *_ = handler_fixture(method="<User:ssfpCSI>")
    handler.method["AverageList"] = np.ones(16, dtype=int)
    raw = BrukerCsiRecoStrategy().sort_kspace(
        handler, data_type="fid", repeated_positions=rule
    )
    trace = json.loads(raw.attrs["csi_raw_trace"])
    assert trace["repeated_positions"] == f"{rule}_with_acquisition_count"
    accepted, reason, _ = audit_csi_raw_input(raw)
    assert (reason == "summed_repeated_positions") is reason_is_summed
    if reason_is_summed:
        assert not accepted
