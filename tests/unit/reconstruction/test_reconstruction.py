import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from test_csi_raw import describe_from_parameters

from hypermri.reconstruction.precision import convert_for_storage
from hypermri.reconstruction.selection import RECONSTRUCTIONS, find_reconstruction
from hypermri.reconstruction.strategies._spectroscopy import FidSpectroRecoStrategy
from hypermri.reconstruction.strategies.agilent import AgilentSpectroRecoStrategy
from hypermri.reconstruction.strategies.bruker import epi as epi_module
from hypermri.reconstruction.strategies.bruker import raw as raw_module
from hypermri.reconstruction.strategies.bruker.bssfp import (
    BrukerBssfpRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.cartesian import (
    BrukerCartesianRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.csi import (
    BrukerCsiRecoStrategy,
    _encoding_order_name,
)
from hypermri.reconstruction.strategies.bruker.dti import BrukerDtiRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.epi import BrukerEpiRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.igflash import (
    BrukerIgflashRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.rare import BrukerRareRawRecoStrategy
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)
from hypermri.reconstruction.strategies.bruker.seq2d import Brukerseq2dRecoStrategy
from hypermri.reconstruction.strategies.bruker.spectroscopy import (
    BrukerSpectroRecoStrategy,
)
from hypermri.reconstruction.strategies.magritek import MagritekSpectroRecoStrategy
from hypermri.utils.coordinates.image_shifts import shift_image


def test_exact_method_strategy_options_override_sequence_alias():
    _, exact_options = find_reconstruction(
        "bruker", "<User:fisp_grad_1metab_CS>", "raw"
    )
    _, alias_options = find_reconstruction(
        "bruker", "<User:fisp_grad_1metab_CS2>", "raw"
    )

    assert exact_options["fov_shift_sign"] == -1.0
    assert exact_options["data_precision"] == "float32"
    assert "fov_shift_sign" not in alias_options
    assert alias_options["data_precision"] == "float32"


def test_unknown_method_uses_the_vendor_default_strategy():
    strategy, options = find_reconstruction("bruker", "unregistered_method", "raw")

    assert strategy is BrukerRawRecoStrategy
    assert options == {"data_precision": "float32"}


def test_specialized_bruker_cartesian_strategies_are_registered():
    igflash, igflash_options = find_reconstruction("bruker", "<Bruker:IgFLASH>", "raw")
    bssfp, bssfp_options = find_reconstruction("bruker", "<Bruker:FISP>", "raw")
    dti, dti_options = find_reconstruction("bruker", "<Bruker:DtiStandard>", "raw")
    epi, epi_options = find_reconstruction("bruker", "<Bruker:EPI>", "raw")
    dtiepi, dtiepi_options = find_reconstruction("bruker", "<Bruker:DtiEpi>", "raw")

    assert igflash is BrukerIgflashRawRecoStrategy
    assert bssfp is BrukerBssfpRawRecoStrategy
    assert dti is BrukerDtiRawRecoStrategy
    assert epi is BrukerEpiRawRecoStrategy
    assert dtiepi is BrukerEpiRawRecoStrategy
    assert igflash_options == {"data_precision": "float32"}
    assert bssfp_options == {"data_precision": "float32"}
    assert dti_options == {"data_precision": "float32"}
    assert epi_options == {"data_precision": "float32"}
    assert dtiepi_options == {"data_precision": "float32"}


def test_bssfp_raw_maps_acquired_objects_to_movie_and_slice(monkeypatch, tmp_path):
    (tmp_path / "fid").write_bytes(b"fid")
    raw = np.zeros((2, 2, 4, 1, 1), dtype=np.complex64)
    for object_index in range(4):
        raw[:, :, object_index, 0, 0] = object_index + 1
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.bssfp.Dataset",
        lambda _path: SimpleNamespace(data=raw),
    )
    strategy = BrukerBssfpRawRecoStrategy()
    monkeypatch.setattr(
        strategy,
        "_finish_reconstruction",
        lambda _handler, kspace, dims, **_kwargs: (kspace, dims),
    )
    handler = SimpleNamespace(
        path=tmp_path,
        matrix=np.array([2, 2, 2]),
        n_slices=2,
        method={"PVM_EncSteps1": np.array([-1, 0]), "PVM_NMovieFrames": 2},
        acqp={"ACQ_obj_order": np.array([0, 1, 2, 3])},
    )

    kspace, dims = strategy.reconstruct(handler)

    assert kspace.shape == (2, 2, 2, 2, 1, 1)
    assert dims == [
        "read_idx",
        "phase_idx",
        "acquisition_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    ]
    np.testing.assert_array_equal(kspace[:, :, 0, 0, 0, 0], 1)
    np.testing.assert_array_equal(kspace[:, :, 1, 0, 0, 0], 2)
    np.testing.assert_array_equal(kspace[:, :, 0, 1, 0, 0], 3)
    np.testing.assert_array_equal(kspace[:, :, 1, 1, 0, 0], 4)


def test_dti_raw_maps_slice_inside_diffusion_object_loop(monkeypatch, tmp_path):
    (tmp_path / "fid").write_bytes(b"fid")
    raw = np.zeros((2, 2, 6, 1, 1), dtype=np.complex64)
    for object_index in range(6):
        raw[:, :, object_index, 0, 0] = object_index + 1
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.dti.Dataset",
        lambda _path: SimpleNamespace(data=raw),
    )
    strategy = BrukerDtiRawRecoStrategy()
    monkeypatch.setattr(
        strategy,
        "_finish_reconstruction",
        lambda _handler, kspace, dims, **_kwargs: (kspace, dims),
    )
    handler = SimpleNamespace(
        path=tmp_path,
        spatial_dimension="2D",
        matrix=np.array([2, 2, 2]),
        n_slices=2,
        method={"PVM_EncSteps1": np.array([-1, 0]), "PVM_DwNDiffExp": 3},
        acqp={"ACQ_obj_order": np.arange(6)},
    )

    kspace, dims = strategy.reconstruct(handler)

    assert kspace.shape == (2, 2, 2, 3, 1, 1)
    assert dims == [
        "read_idx",
        "phase_idx",
        "slice_idx",
        "bvalue_idx",
        "repetition_idx",
        "channel_idx",
    ]
    np.testing.assert_array_equal(kspace[:, :, 0, 0, 0, 0], 1)
    np.testing.assert_array_equal(kspace[:, :, 1, 0, 0, 0], 2)
    np.testing.assert_array_equal(kspace[:, :, 0, 1, 0, 0], 3)
    np.testing.assert_array_equal(kspace[:, :, 1, 2, 0, 0], 6)


@pytest.mark.parametrize("reader_mirrored", [False, True])
def test_epi_raw_uses_shot_local_echo_polarity(monkeypatch, reader_mirrored):
    read_size, phase_size, echoes_per_shot = 4, 6, 3
    steps = np.array([-3, -1, 1, -2, 0, 2])
    acquired = np.zeros((read_size, phase_size), dtype=np.complex64)
    for line in range(phase_size):
        acquired[:, line] = 10 * line + np.arange(read_size)

    phase_order = np.argsort(steps)
    brukerapi_data = acquired[:, phase_order].copy()
    if reader_mirrored:
        brukerapi_data[:, 1::2] = brukerapi_data[::-1, 1::2]
    brukerapi_data = brukerapi_data[:, :, None, None, None]
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.epi.Dataset",
        lambda _path: SimpleNamespace(
            data=brukerapi_data,
            _schema=SimpleNamespace(mirror_odd_lines=reader_mirrored),
        ),
    )
    handler = SimpleNamespace(
        path=Path("/unused"),
        matrix=np.array([read_size, phase_size, 1]),
        n_slices=1,
        n_echoes=1,
        n_inversion_time=1,
        b_values=np.array([0.0]),
        method={
            "Method": "<Bruker:EPI>",
            "PVM_EncSteps1": steps,
            "PVM_EpiNEchoes": echoes_per_shot,
            "PVM_EpiNShots": 2,
        },
        acqp={"ACQ_obj_order": 0},
    )

    kspace, dims, attrs = BrukerEpiRawRecoStrategy()._reconstruct_2d(
        handler, {"apply_phase_correction": False}
    )

    expected = acquired.copy()
    odd_acquired_lines = np.flatnonzero((np.arange(phase_size) % echoes_per_shot) % 2)
    expected[:, odd_acquired_lines] = expected[::-1, odd_acquired_lines]
    expected = expected[:, phase_order]
    np.testing.assert_array_equal(kspace[:, :, 0, 0], expected)
    assert dims == ["read_idx", "phase_idx", "slice_idx", "channel_idx"]
    assert attrs["epi_shots"] == 2


def test_epi_raw_regrids_measured_read_trajectory():
    trajectory = np.array([0.0, 0.5, 2.0, 3.5, 4.0])
    profile = trajectory**3 + 2j * trajectory**2
    values = np.stack((profile, 2 * profile), axis=1).astype(np.complex64)
    method = {
        "PVM_EpiTrajAdjYesNo": "Yes",
        "PVM_EpiTrajAdjMeasured": "Yes",
        "PVM_EpiTrajAdjMatrix0": 5,
        "PVM_EpiTrajAdjkx": trajectory,
        "PVM_EpiTrajAdjb0": np.zeros(5),
    }

    regridded, attrs = BrukerEpiRawRecoStrategy()._apply_trajectory_regridding(
        values,
        method,
        {},
        axis=0,
    )

    target = np.arange(5, dtype=float)
    expected = target**3 + 2j * target**2
    np.testing.assert_allclose(regridded[:, 0], expected, atol=1e-5)
    np.testing.assert_allclose(regridded[:, 1], 2 * expected, atol=1e-5)
    assert attrs["epi_trajectory_regridding"] is True
    assert attrs["epi_trajectory_source"] == "PVM_EpiTrajAdjkx"
    assert attrs["epi_b0_trajectory_available"] is True
    assert attrs["epi_b0_trajectory_correction"] is False


def test_epi_raw_regridding_rejects_incomplete_measured_trajectory():
    values = np.zeros((5, 2), dtype=np.complex64)
    method = {
        "PVM_EpiTrajAdjYesNo": "Yes",
        "PVM_EpiTrajAdjMeasured": "Yes",
        "PVM_EpiTrajAdjMatrix0": 5,
        "PVM_EpiTrajAdjkx": np.arange(4),
    }

    with np.testing.assert_raises_regex(
        ValueError,
        "expected 5 finite values",
    ):
        BrukerEpiRawRecoStrategy()._apply_trajectory_regridding(
            values,
            method,
            {},
            axis=0,
        )


def test_epi_phase_correction_uses_centered_read_coordinate(monkeypatch):
    strategy = BrukerEpiRawRecoStrategy()
    monkeypatch.setattr(epi_module, "centered_ifft", lambda values, _axes: values)
    monkeypatch.setattr(epi_module, "centered_fft", lambda values, _axes: values)
    kspace = np.ones((4, 2, 1, 1, 1), dtype=np.complex64)

    corrected = strategy._apply_phase_correction(
        kspace,
        np.array([1]),
        np.array([[0.0, 0.125]]),
    )

    expected_phase = np.exp(2j * np.pi * 0.125 * (np.arange(4) - 2))
    np.testing.assert_allclose(corrected[:, 1, 0, 0, 0], expected_phase)
    np.testing.assert_array_equal(corrected[:, 0, 0, 0, 0], 1)


def test_igflash_raw_bins_repeated_profiles_without_fid(monkeypatch, tmp_path):
    (tmp_path / "rawdata.job0").write_bytes(b"raw")
    read_size, phase_size, slices = 2, 4, 2
    segments, oversamples, movie_frames = 2, 4, 2
    lines = phase_size // segments
    acquired = np.zeros(
        (read_size, segments, oversamples, lines, slices), dtype=np.complex64
    )
    for segment in range(segments):
        for oversample in range(oversamples):
            for line in range(lines):
                for slice_index in range(slices):
                    acquired[:, segment, oversample, line, slice_index] = (
                        1000 * segment + 100 * oversample + 10 * line + slice_index
                    )
    raw = acquired.transpose(1, 2, 3, 4, 0).reshape(-1)
    strategy = BrukerIgflashRawRecoStrategy()
    monkeypatch.setattr(
        strategy,
        "_finish_reconstruction",
        lambda _handler, kspace, dims, **_kwargs: (kspace, dims),
    )
    handler = SimpleNamespace(
        path=tmp_path,
        matrix=np.array([read_size, phase_size, slices]),
        n_slices=slices,
        rawdatajob0=raw,
        method={
            "Nsegments": segments,
            "PVM_NMovieFrames": movie_frames,
            "PVM_EncSteps1": np.array([-2, 0, -1, 1]),
        },
        acqp={"ACQ_obj_order": np.array([1, 0])},
    )

    kspace, dims = strategy.reconstruct(handler)

    assert kspace.shape == (read_size, phase_size, movie_frames, slices, 1)
    assert dims == [
        "read_idx",
        "phase_idx",
        "acquisition_idx",
        "slice_idx",
        "channel_idx",
    ]
    # Movie 0 averages oversamples 0 and 2; acquired slice 1 maps to physical 0.
    np.testing.assert_array_equal(kspace[:, 0, 0, 0, 0], 101)
    # Movie 1 averages oversamples 1 and 3.
    np.testing.assert_array_equal(kspace[:, 0, 1, 0, 0], 201)


def test_raw_precision_conversion_does_not_touch_scanner_reconstruction():
    handler = SimpleNamespace(
        fid=np.ones(2, dtype=np.complex128),
        seq2d=np.array([100_000.0], dtype=np.float32),
    )

    fid = BrukerRawRecoStrategy._fid_for_calculation(handler, True, np.float16)

    assert fid.dtype == np.complex64
    assert handler.fid.dtype == np.complex128
    assert handler.seq2d.dtype == np.float32
    assert np.isfinite(handler.seq2d).all()


@pytest.mark.parametrize(
    "sequence",
    ["flash", "<User:fisp_grad_1metab_CS>", "rare", "default"],
)
def test_bruker_raw_reconstruction_config_uses_float32(sequence):
    """Configured Bruker raw reconstructions never use float16."""
    _, options = find_reconstruction("bruker", sequence, "raw")

    assert options["data_precision"] == "float32"


@pytest.mark.parametrize(
    "strategy_class",
    [BrukerRawRecoStrategy, BrukerMGERawRecoStrategy, BrukerRareRawRecoStrategy],
)
def test_bruker_raw_reconstruction_default_precision_is_float32(strategy_class):
    """Without a ``data_precision`` option the raw path keeps float32 data.

    float16 overflows above 65504 and keeps about three significant digits,
    which is below the dynamic range of Bruker raw intensities.
    """
    strategy = strategy_class()
    captured = {}

    def record_precision(**kwargs):
        captured["data_precision"] = kwargs.get("data_precision", "not passed")

    strategy.reco_rawdata = record_precision
    strategy.postprocess_fid_rare = lambda handler, data_precision: handler.fid
    fid = np.array([1.0 + 1.0e-4j, 2.0e5 + 0.0j], dtype=np.complex128)
    handler = SimpleNamespace(fid=fid.copy(), expnum=1)

    strategy.reconstruct(handler, append_parameters=False)

    assert captured["data_precision"] is np.float32
    # The acquired complex FID is not downcast before the Fourier transform.
    assert handler.fid.dtype == np.complex128
    np.testing.assert_array_equal(handler.fid, fid)
    assert convert_for_storage(
        np.array([7.0e4], dtype=np.float64), captured["data_precision"]
    ) == pytest.approx(7.0e4, rel=1e-6)


def test_dti_alias_and_canonical_key_share_specialized_strategy():
    canonical_strategy, canonical_options = find_reconstruction("bruker", "dti", "raw")
    alias_strategy, alias_options = find_reconstruction(
        "bruker", "<Bruker:DtiStandard>", "raw"
    )

    assert canonical_strategy is BrukerDtiRawRecoStrategy
    assert alias_strategy is BrukerDtiRawRecoStrategy
    assert canonical_options == {"data_precision": "float32"}
    assert alias_options == canonical_options


def test_mge_raw_strategy_is_selected_for_mge():
    canonical_strategy, canonical_options = find_reconstruction("bruker", "mge", "raw")
    alias_strategy, alias_options = find_reconstruction("bruker", "<Bruker:MGE>", "raw")

    assert canonical_strategy is BrukerMGERawRecoStrategy
    assert alias_strategy is BrukerMGERawRecoStrategy
    assert canonical_options == {}
    assert alias_options == canonical_options


def test_shift_image_fov_shift_sign_inverts_automatic_phase_shift():
    kspace = xr.DataArray(
        np.ones((4, 4), dtype=np.complex64),
        dims=("read_idx", "phase_idx"),
        attrs={"domain": "kspace", "spatial_dimension": "2D", "num_image_fov_shift": 0},
    )
    data_obj = SimpleNamespace(
        type="bruker",
        read_offset_mm=0.0,
        phase_offset_mm=1.0,
        slice_offset_mm=0.0,
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        patient_pos="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
        spatial_dimension="2D",
    )

    shifted_default = shift_image(
        data=kspace.copy(deep=True),
        scan=data_obj,
        shift_method="phase",
        domain="kspace",
        fov_shift_sign=1.0,
    )
    shifted_inverted = shift_image(
        data=kspace.copy(deep=True),
        scan=data_obj,
        shift_method="phase",
        domain="kspace",
        fov_shift_sign=-1.0,
    )

    assert shifted_default.attrs["num_image_fov_shift"] == 1
    assert shifted_inverted.attrs["num_image_fov_shift"] == 1
    assert not np.allclose(shifted_default.values, shifted_inverted.values)
    np.testing.assert_allclose(
        shifted_default.values,
        np.conj(shifted_inverted.values),
        atol=1e-7,
    )


def test_bruker_raw_reconstruct_forwards_fov_shift_sign(monkeypatch):
    captured = {}

    def fake_reco_rawdata(self, **kwargs):
        captured.update(kwargs)
        return xr.DataArray(
            np.zeros((1, 1), dtype=np.complex64),
            dims=("read_idx", "phase_idx"),
            attrs={"domain": "spatial"},
        )

    monkeypatch.setattr(BrukerRawRecoStrategy, "reco_rawdata", fake_reco_rawdata)

    strategy = BrukerRawRecoStrategy()
    handler = SimpleNamespace(fid=np.ones(2), fid_pp=np.ones(2))

    strategy.reconstruct(handler, fov_shift_sign=-1.0, data_precision="float32")

    assert captured["fov_shift_sign"] == -1.0


def test_bruker_raw_2d_multislice_reshape_uses_phase_major_storage():
    read_size, phase_size, slice_size = 3, 2, 3
    fid = np.zeros((phase_size * slice_size, read_size), dtype=np.complex64)
    for phase_idx in range(phase_size):
        for slice_idx in range(slice_size):
            row = phase_idx * slice_size + slice_idx
            fid[row, :] = 100 * phase_idx + 10 * slice_idx + np.arange(read_size)

    handler = SimpleNamespace(
        n_repetitions=1,
        enc_matrix=np.array([read_size, phase_size, slice_size]),
        matrix=np.array([read_size, phase_size, slice_size]),
        n_receivers=1,
        spatial_dimension="2D",
        method={"PVM_EncSteps1": np.arange(phase_size) - phase_size // 2},
    )

    kspace = BrukerRawRecoStrategy().reshape_fid_to_kspace(fid, handler)

    assert kspace.shape == (1, 1, slice_size, phase_size, 1, read_size)
    for slice_idx in range(slice_size):
        for phase_idx in range(phase_size):
            expected = 100 * phase_idx + 10 * slice_idx + np.arange(read_size)
            np.testing.assert_array_equal(
                kspace[0, 0, slice_idx, phase_idx, 0, :],
                expected,
            )


def test_csi_normalizes_list_encoding_order():
    assert _encoding_order_name(["LINEAR_ENC", "LINEAR_ENC"]) == "LINEAR_ENC LINEAR_ENC"


def test_csi_reconstruct_uses_specialized_spatial_ifft(monkeypatch):
    strategy = BrukerCsiRecoStrategy()
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.csi.shift_image",
        lambda **kwargs: kwargs["data"],
    )
    kspace = xr.DataArray(
        np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2),
        dims=(
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ),
        attrs={"domain": "kspace", "signal_domain": "temporal"},
    )
    handler = SimpleNamespace(
        fid=np.ones(4, dtype=np.complex64),
        is_spectroscopic=True,
        read_offset_mm=0.0,
        phase_offset_mm=0.0,
        slice_offset_mm=0.0,
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        type="CSI",
        get=lambda _name, default=None: default,
    )

    monkeypatch.setattr(strategy, "sort_kspace", lambda **_kwargs: kspace)
    monkeypatch.setattr(
        BrukerCsiRecoStrategy,
        "reco_rawdata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )

    reco_data = strategy.reconstruct(
        handler, append_parameters=False, data_precision="float32"
    )

    assert reco_data.dims == (
        "acquisition_idx",
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    )
    assert reco_data.attrs["domain"] == "spatial"
    assert reco_data.attrs["signal_domain"] == "temporal"
    assert np.iscomplexobj(reco_data.values)


def test_csi_sorted_data_spatially_reconstructs_sorted_grid(monkeypatch):
    strategy = BrukerCsiRecoStrategy()
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.csi.shift_image",
        lambda **kwargs: kwargs["data"],
    )
    kspace = xr.DataArray(
        np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2),
        dims=(
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ),
        attrs={"domain": "kspace", "signal_domain": "temporal"},
    )
    handler = SimpleNamespace()

    reco_data = strategy._sorted_csi_to_data(kspace, handler, data_precision=np.float32)

    expected = np.transpose(kspace.values, (0, 5, 3, 2, 1, 4))
    for axis in (1, 2):
        expected = np.fft.ifftshift(
            np.fft.ifft(
                np.fft.ifftshift(expected, axes=(axis,)),
                axis=axis,
            ),
            axes=(axis,),
        )
    np.testing.assert_allclose(reco_data.values, expected.astype(np.complex64))
    assert reco_data.dtype == np.complex64
    assert reco_data.attrs["domain"] == "spatial"
    assert reco_data.attrs["signal_domain"] == "temporal"


def test_stock_bruker_csi_reco_keeps_scanner_spatial_grid_without_spatial_ifft():
    strategy = BrukerCsiRecoStrategy()
    kspace = xr.DataArray(
        np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2),
        dims=(
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ),
        attrs={"domain": "kspace", "signal_domain": "temporal"},
    )
    handler = SimpleNamespace(method={"Method": "<Bruker:CSI>"})

    reco_data = strategy._sorted_csi_to_data(kspace, handler, data_precision=np.float32)

    expected = np.transpose(kspace.values, (0, 5, 3, 2, 1, 4))
    np.testing.assert_array_equal(reco_data.values, expected.astype(np.complex64))
    assert reco_data.dtype == np.complex64
    assert reco_data.attrs["domain"] == "spatial"
    assert reco_data.attrs["signal_domain"] == "temporal"
    assert reco_data.attrs["csi_spatial_reconstruction"] == "scanner_grid"


def test_csi_linear_sort_preserves_phase_read_order_without_roll():
    strategy = BrukerCsiRecoStrategy()
    handler = SimpleNamespace(
        fid=np.array([1, 2, 3, 4], dtype=np.complex64),
        rawdatajob0=np.array([], dtype=np.complex64),
        n_repetitions=1,
        n_slices=1,
        matrix=np.array([2, 2, 1]),
        n_acquisition=1,
        n_receivers=1,
        method={
            "Method": "<Bruker:CSI>",
            "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
            "PVM_EncSteps0": np.array([-1, 0]),
            "PVM_EncSteps1": np.array([-1, 0]),
            "PVM_Matrix": np.array([2, 2, 1]),
        },
        read_orientation="L_R",
        sampling_time_axis_s=np.array([0.0]),
        rps_grid_mm=np.zeros((3, 2, 2, 1)),
        patient_pos="Head_Prone",
        slice_orientation="axial",
        acquisition_matrix=np.array([2, 2, 1]),
        slice_thickness_mm=1.0,
        git_commit_hash=None,
        spatial_dimension="2D",
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        rps_extent_mm=np.array([2.0, 2.0, 1.0]),
        rps_axes_mm=[
            np.array([-1.0, 1.0]),
            np.array([-1.0, 1.0]),
            np.array([0.0]),
        ],
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    handler.method_parameter = lambda name: handler.method[name]
    handler.to_csi_acquisition = lambda: describe_from_parameters(handler)

    kspace = strategy.sort_kspace(handler, data_type="fid", write_to_object=False)

    sorted_grid = kspace.isel(
        acquisition_idx=0, repetition_idx=0, slice_idx=0, channel_idx=0
    ).values
    np.testing.assert_array_equal(sorted_grid, np.array([[1, 2], [3, 4]]))


def _linear_stock_csi_handler(
    *,
    n_acquisition: int = 6,
    cut_off: int = 2,
    dwell_time_s: float = 5e-4,
) -> SimpleNamespace:
    """Build a small stock-CSI handler whose acquisition points are traceable."""
    n_read = 2
    n_phase = 2
    payload = np.zeros(
        (1, 1, n_phase, n_read, 1, n_acquisition),
        dtype=np.complex64,
    )
    for phase_idx in range(n_phase):
        for read_idx in range(n_read):
            payload[0, 0, phase_idx, read_idx, 0, :] = (
                100 * phase_idx + 10 * read_idx + np.arange(n_acquisition)
            )

    handler = SimpleNamespace(
        fid=payload.reshape(-1),
        rawdatajob0=np.array([], dtype=np.complex64),
        n_repetitions=1,
        n_slices=1,
        matrix=np.array([n_read, n_phase, 1]),
        n_acquisition=n_acquisition,
        n_receivers=1,
        cut_off=cut_off,
        dwell_time_s=dwell_time_s,
        method={
            "Method": "<Bruker:CSI>",
            "PVM_EncOrder": ["LINEAR_ENC", "LINEAR_ENC"],
            "PVM_EncSteps0": np.array([-1, 0]),
            "PVM_EncSteps1": np.array([-1, 0]),
            "PVM_Matrix": np.array([n_read, n_phase, 1]),
        },
        read_orientation="L_R",
        sampling_time_axis_s=np.arange(n_acquisition, dtype=float) * dwell_time_s,
        freq_axis_ppm=None,
        freq_axis_hz=None,
        rps_grid_mm=np.zeros((3, n_read, n_phase, 1)),
        patient_pos="Head_Prone",
        patient_position="Head_Prone",
        slice_orientation="axial",
        acquisition_matrix=np.array([n_read, n_phase, 1]),
        slice_thickness_mm=1.0,
        git_commit_hash=None,
        spatial_dimension="2D",
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        rps_extent_mm=np.array([2.0, 2.0, 1.0]),
        rps_axes_mm=[
            np.array([-1.0, 1.0]),
            np.array([-1.0, 1.0]),
            np.array([0.0]),
        ],
        is_spectroscopic=True,
        expnum=1,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    handler.method_parameter = lambda name: handler.method[name]
    handler.to_csi_acquisition = lambda: describe_from_parameters(handler)
    return handler


def test_csi_reconstruct_applies_handler_cutoff_once_and_rebases_time_axis():
    strategy = BrukerCsiRecoStrategy()
    handler = _linear_stock_csi_handler()

    cut = strategy.reconstruct(
        handler,
        append_parameters=True,
        data_precision="float32",
    )
    full = strategy.reconstruct(
        _linear_stock_csi_handler(),
        cut_off=0,
        append_parameters=True,
        data_precision="float32",
    )

    assert cut.dims == (
        "acquisition_idx",
        "read_idx",
        "phase_idx",
        "slice_idx",
        "repetition_idx",
        "channel_idx",
    )
    assert cut.shape == (4, 2, 2, 1, 1, 1)
    assert full.shape == (6, 2, 2, 1, 1, 1)
    np.testing.assert_array_equal(
        cut.values,
        full.isel(acquisition_idx=slice(handler.cut_off, None)).values,
    )

    expected_time_s = np.arange(4, dtype=float) * handler.dwell_time_s
    for coord_name in ("acquisition_idx", "acquisition_time_s"):
        assert coord_name in cut.coords
        np.testing.assert_allclose(
            np.asarray(cut.coords[coord_name]),
            expected_time_s,
            rtol=0.0,
            atol=1e-15,
        )
    assert cut.coords["acquisition_time_s"].attrs["unit"] == "s"
    np.testing.assert_allclose(
        np.asarray(cut.attrs["sampling_time_axis_s"]),
        expected_time_s,
        rtol=0.0,
        atol=1e-15,
    )

    # Reconstruction is a derived view; the acquired handler payload remains intact.
    assert handler.n_acquisition == 6
    assert handler.fid.size == 24
    assert len(handler.sampling_time_axis_s) == 6


def test_csi_reconstruct_explicit_zero_cutoff_overrides_handler_default():
    strategy = BrukerCsiRecoStrategy()
    handler = _linear_stock_csi_handler(cut_off=2)

    full = strategy.reconstruct(
        handler,
        cut_off=0,
        append_parameters=True,
        data_precision="float32",
    )

    assert full.sizes["acquisition_idx"] == handler.n_acquisition == 6
    expected_time_s = np.arange(6, dtype=float) * handler.dwell_time_s
    np.testing.assert_allclose(
        np.asarray(full.coords["acquisition_time_s"]),
        expected_time_s,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(full.attrs["sampling_time_axis_s"]),
        expected_time_s,
        rtol=0.0,
        atol=1e-15,
    )


def test_spectroscopic_seq2d_is_marked_as_spectral_domain():
    strategy = Brukerseq2dRecoStrategy()
    handler = SimpleNamespace(
        is_spectroscopic=False,
        type="<User:lucaCSI4>",
        sequence_name="<User:lucaCSI4>",
        method={"Method": "<User:lucaCSI4>", "PVM_SpecMatrix": 4},
        patient_pos="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
        acquisition_matrix=np.eye(3),
        n_slices=1,
        git_commit_hash=None,
        matrix=np.array([2, 2, 1]),
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        rps_extent_mm=np.array([2.0, 2.0, 1.0]),
        rps_axes_mm=[
            np.array([-0.5, 0.5]),
            np.array([-0.5, 0.5]),
            np.array([0.0]),
        ],
        freq_axis_ppm=np.array([5.0, 4.0, 3.0, 2.0]),
        freq_axis_hz=np.array([-200.0, -100.0, 0.0, 100.0]),
        sampling_time_axis_s=None,
        frame_time_s=None,
        expnum=1,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    data = np.ones((4, 2, 2), dtype=np.float32)

    result = strategy.set_scanner_data(
        data=data,
        dims=("acquisition_idx", "read_idx", "phase_idx"),
        handler=handler,
    )

    assert result.attrs["domain"] == "spatial"
    assert result.attrs["signal_domain"] == "spectral"
    np.testing.assert_array_equal(result["freq_axis_ppm"], handler.freq_axis_ppm)
    np.testing.assert_array_equal(result["freq_axis_hz"], handler.freq_axis_hz)
    assert result["freq_axis_hz"].dims == ("acquisition_idx",)
    assert not {"freq_axis_hz", "freq_axis_ppm"} & set(result.attrs)


def test_epsi_seq2d_is_marked_as_spectral_domain_from_spec_size():
    strategy = Brukerseq2dRecoStrategy()
    handler = SimpleNamespace(
        is_spectroscopic=False,
        type="<Bruker:EPSI>",
        sequence_name="<Bruker:EPSI>",
        method={"Method": "<Bruker:EPSI>", "SpecSize": 4},
        patient_pos="Head_Prone",
        slice_orientation="axial",
        read_orientation="L_R",
        acquisition_matrix=np.eye(3),
        n_slices=1,
        git_commit_hash=None,
        matrix=np.array([2, 3, 1]),
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        rps_extent_mm=np.array([2.0, 3.0, 1.0]),
        rps_axes_mm=[
            np.array([-0.5, 0.5]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.0]),
        ],
        freq_axis_ppm=np.array([5.0, 4.0, 3.0, 2.0]),
        freq_axis_hz=np.array([-200.0, -100.0, 0.0, 100.0]),
        sampling_time_axis_s=None,
        frame_time_s=None,
        expnum=1,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    data = np.ones((2, 4, 3), dtype=np.float32)

    result = strategy.set_scanner_data(
        data=data,
        dims=("read_idx", "acquisition_idx", "phase_idx"),
        handler=handler,
    )

    assert result.attrs["domain"] == "spatial"
    assert result.attrs["signal_domain"] == "spectral"


def test_bruker_spectro_reorder_preserves_averaged_fid_length():
    handler = SimpleNamespace(
        fid=np.ones(2 * 2048, dtype=np.complex64),
        n_repetitions=2,
        n_averages=64,
        n_slices=1,
        n_receivers=1,
        n_acquisition=2048,
        method={},
        reco={},
    )

    fid = BrukerSpectroRecoStrategy().reorder_reshape_fid(handler)

    assert fid.shape == (2048, 2, 1, 1, 1, 1)
    assert handler.n_acquisition == 2048


def test_bruker_spectro_reorder_keeps_cpmg_echo_axis():
    handler = SimpleNamespace(
        fid=np.ones(12 * 256, dtype=np.complex64),
        n_repetitions=1,
        n_averages=5,
        n_slices=1,
        n_receivers=1,
        n_acquisition=256,
        method={"NEchoes": 12},
        reco={"RecoObjectsPerRepetition": 12},
    )

    fid = BrukerSpectroRecoStrategy().reorder_reshape_fid(handler)

    assert fid.shape == (256, 1, 1, 12, 1, 1)


def _small_spectroscopy_handler(cut_off: int = 2) -> SimpleNamespace:
    dwell_time_s = 0.125
    handler = SimpleNamespace(
        id=1,
        expnum=1,
        fid=(np.arange(6) + 1j * np.arange(6, 12)).astype(np.complex64),
        n_repetitions=1,
        n_averages=1,
        n_slices=1,
        n_receivers=1,
        n_acquisition=6,
        method={},
        reco={},
        cut_off=cut_off,
        dwell_time_s=dwell_time_s,
        sampling_time_axis_s=np.arange(6, dtype=float) * dwell_time_s,
        bandwidth_hz=1.0 / dwell_time_s,
        bandwidth_ppm=4.0,
        center_ppm=0.0,
        freq_axis_ppm=None,
        freq_axis_hz=None,
        is_spectroscopic=True,
    )
    handler.get = lambda name, default=None: getattr(handler, name, default)
    return handler


def _stub_spectroscopy_reco(strategy, monkeypatch) -> None:
    monkeypatch.setattr(
        strategy,
        "reco_rawdata",
        lambda handler, **kwargs: kwargs["kspace_xarray"],
    )


def test_bruker_spectro_reconstruct_aligns_axes_after_leading_cutoff(monkeypatch):
    strategy = BrukerSpectroRecoStrategy()
    _stub_spectroscopy_reco(strategy, monkeypatch)
    handler = _small_spectroscopy_handler()
    original_fid = handler.fid.copy()

    result = strategy.reconstruct(handler, data_precision="float32")

    expected_fid = original_fid[handler.cut_off :]
    expected_spectrum = np.fft.fftshift(np.fft.fft(expected_fid))
    np.testing.assert_allclose(result.values.squeeze(), expected_spectrum)
    expected_time_s = np.arange(4, dtype=float) * handler.dwell_time_s
    for coord_name in ("acquisition_idx", "acquisition_time_s"):
        np.testing.assert_allclose(result.coords[coord_name], expected_time_s)
        assert result.coords[coord_name].attrs["unit"] == "s"
    np.testing.assert_allclose(result.attrs["sampling_time_axis_s"], expected_time_s)
    assert result.sizes["acquisition_idx"] == 4
    assert result.coords["freq_axis_hz"].size == 4
    assert result.coords["freq_axis_ppm"].size == 4
    assert result.attrs["cut_off"] == 2
    assert "applied_cut_off" not in result.attrs
    assert result.attrs["raw_n_acquisition"] == 6
    assert handler.fid.size == 6
    assert handler.n_acquisition == 6
    assert not hasattr(handler, "data")
    assert not hasattr(handler, "fid_pp")


def test_bruker_spectro_reconstruction_is_repeatable_without_mutating_fid(
    monkeypatch,
):
    """Repeated explicit reconstruction must not apply the cutoff cumulatively."""
    strategy = BrukerSpectroRecoStrategy()
    _stub_spectroscopy_reco(strategy, monkeypatch)
    handler = _small_spectroscopy_handler()
    acquired_fid = handler.fid.copy()

    first = strategy.reconstruct(handler, data_precision="float32")
    second = strategy.reconstruct(handler, data_precision="float32")

    xr.testing.assert_identical(first, second)
    np.testing.assert_array_equal(handler.fid, acquired_fid)
    assert first.sizes["acquisition_idx"] == 4
    assert second.sizes["acquisition_idx"] == 4
    assert first.attrs["cut_off"] == 2
    assert second.attrs["cut_off"] == 2


def test_bruker_spectro_explicit_zero_cutoff_overrides_handler_default(monkeypatch):
    strategy = BrukerSpectroRecoStrategy()
    _stub_spectroscopy_reco(strategy, monkeypatch)
    handler = _small_spectroscopy_handler(cut_off=2)

    result = strategy.reconstruct(
        handler,
        cut_off=0,
        data_precision="float32",
    )

    assert result.sizes["acquisition_idx"] == 6
    assert result.attrs["cut_off"] == 0


@pytest.mark.parametrize("cut_off", [-1, 6, 1.5, "invalid"])
def test_bruker_spectro_rejects_invalid_cutoff(cut_off):
    strategy = BrukerSpectroRecoStrategy()
    handler = _small_spectroscopy_handler()
    sorted_fid = handler.fid.reshape((6, 1, 1, 1, 1, 1))

    with pytest.raises(ValueError, match="cut_off"):
        strategy.remove_leading_fid_points(handler, cut_off=cut_off, values=sorted_fid)


def test_bruker_spectro_cutoff_needs_explicit_fid_and_has_no_legacy_alias():
    """The sorted FID is passed explicitly; the misnamed alias is gone."""
    strategy = BrukerSpectroRecoStrategy()
    handler = _small_spectroscopy_handler()
    sorted_fid = handler.fid.reshape((6, 1, 1, 1, 1, 1))

    result = strategy.remove_leading_fid_points(handler, values=sorted_fid)

    np.testing.assert_array_equal(result, sorted_fid[handler.cut_off :])
    with pytest.raises(TypeError, match="values"):
        strategy.remove_leading_fid_points(handler)
    assert not hasattr(BrukerSpectroRecoStrategy, "remove_trailing_zeros")


def test_rare_postprocess_does_not_treat_off_as_fat_water(monkeypatch):
    strategy = BrukerRareRawRecoStrategy()
    monkeypatch.setattr(
        strategy, "postprocess_fid_file", lambda values, handler: values
    )
    monkeypatch.setattr(
        strategy,
        "_get_phase_encode_order",
        lambda handler: (np.arange(handler.enc_matrix[1]), None),
    )

    raw = np.arange(2 * 2 * 1 * 3, dtype=np.complex64).reshape(2, 2, 1, 3)
    handler = SimpleNamespace(
        expnum=1,
        fid=raw.ravel(),
        rare_factor=2,
        n_receivers=1,
        matrix=np.array([3, 2, 2]),
        enc_matrix=np.array([3, 2, 2]),
        n_repetitions=1,
        n_inversion_time=1,
        fat_water_separation="Off",
        spatial_dimension="2D",
    )

    processed = strategy.postprocess_fid_rare(handler)

    assert processed.shape == (1, 1, 2, 2, 1, 3)
    np.testing.assert_array_equal(processed[0, 0, 0, 0, 0, :], raw[0, 0, 0, :])
    np.testing.assert_array_equal(processed[0, 0, 0, 1, 0, :], raw[0, 1, 0, :])


def test_rare_postprocess_splits_waterfat_from_image_loop(monkeypatch):
    strategy = BrukerRareRawRecoStrategy()
    monkeypatch.setattr(
        strategy, "postprocess_fid_file", lambda values, handler: values
    )
    monkeypatch.setattr(
        strategy,
        "_get_phase_encode_order",
        lambda handler: (np.arange(handler.enc_matrix[1]), None),
    )

    raw = np.zeros((2, 2, 2, 2, 1, 3), dtype=np.complex64)
    for phase_segment_idx in range(raw.shape[0]):
        for waterfat_idx in range(raw.shape[1]):
            for slice_idx in range(raw.shape[2]):
                for echo_idx in range(raw.shape[3]):
                    raw[phase_segment_idx, waterfat_idx, slice_idx, echo_idx, 0, :] = (
                        phase_segment_idx * 1000
                        + waterfat_idx * 100
                        + slice_idx * 10
                        + echo_idx
                    )

    handler = SimpleNamespace(
        expnum=1,
        fid=raw.ravel(),
        rare_factor=2,
        n_receivers=1,
        matrix=np.array([3, 4, 2]),
        enc_matrix=np.array([3, 4, 2]),
        n_repetitions=1,
        n_inversion_time=1,
        fat_water_separation="On",
        spatial_dimension="2D",
    )

    processed = strategy.postprocess_fid_rare(handler)

    assert processed.shape == (1, 1, 2, 4, 2, 1, 3)
    np.testing.assert_array_equal(processed[0, 0, 0, 0, 0, 0, :], raw[0, 0, 0, 0, 0, :])
    np.testing.assert_array_equal(processed[0, 0, 0, 1, 0, 0, :], raw[0, 0, 0, 1, 0, :])
    np.testing.assert_array_equal(processed[0, 0, 0, 2, 0, 0, :], raw[1, 0, 0, 0, 0, :])
    np.testing.assert_array_equal(processed[0, 0, 0, 0, 1, 0, :], raw[0, 1, 0, 0, 0, :])
    np.testing.assert_array_equal(processed[0, 0, 1, 3, 1, 0, :], raw[1, 1, 1, 1, 0, :])


def _synthetic_raw_handler():
    """Return a synthetic 3 x 2 x 2 k-space handler without FOV offsets.

    The geometry values are arbitrary; no scanner data are used.
    """

    class Handler(SimpleNamespace):
        def get(self, name, default=None):
            return getattr(self, name, default)

    read_size, phase_size, slice_size = 3, 2, 2
    rps_grid_mm = np.meshgrid(
        np.arange(read_size),
        np.arange(phase_size),
        np.arange(slice_size),
        indexing="ij",
    )
    kspace = np.ones(
        (1, 1, slice_size, phase_size, 2, 1, read_size), dtype=np.complex64
    ) * (1.0 + 1.0j)
    handler = Handler(
        expnum=1,
        fid_pp=kspace,
        sampling_time_axis_s=np.array([0.0]),
        n_inversion_time=1,
        fat_water_separation="On",
        n_receivers=1,
        rps_grid_mm=rps_grid_mm,
        n_slices=slice_size,
        slice_order=np.arange(slice_size),
        patient_pos="Foot_Supine",
        slice_orientation="axial",
        read_orientation="L_R",
        acquisition_matrix=np.array([1, 1, 1]),
        slice_thickness_mm=1.0,
        git_commit_hash=None,
        spatial_dimension="2D",
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        frame_time_s=None,
    )
    return handler


def test_raw_reco_labels_rare_waterfat_axis():
    handler = _synthetic_raw_handler()

    reco = BrukerRawRecoStrategy().reco_rawdata(
        handler=handler,
        values=handler.fid_pp,
        postprocess_fid=False,
        reshape_fid=False,
        set_kspace_xarray=True,
        shift_xarray=False,
        set_data=True,
        data_precision=np.float32,
    )

    assert "waterfat_idx" in reco.dims
    assert reco.sizes["waterfat_idx"] == 2
    assert list(reco.coords["waterfat_idx"].values) == ["Water", "Fat"]


# Attribute names removed on 15 November 2026 because a second name for the
# same quantity can diverge after slicing or processing.
REMOVED_DUPLICATE_ATTRS = {
    "applied_cut_off",
    "spat_dim",
    "freq_range_hz",
    "freq_range_Hz",
    "freq_range_ppm",
}


def test_raw_reco_result_names_the_spatial_dimension_once():
    handler = _synthetic_raw_handler()

    reco = BrukerRawRecoStrategy().reco_rawdata(
        handler=handler,
        values=handler.fid_pp,
        postprocess_fid=False,
        reshape_fid=False,
        set_kspace_xarray=True,
        shift_xarray=False,
        set_data=True,
        data_precision=np.float32,
    )

    assert reco.attrs["spatial_dimension"] == "2D"
    assert not REMOVED_DUPLICATE_ATTRS & set(reco.attrs)


def test_cartesian_base_attrs_name_the_spatial_dimension_once():
    handler = SimpleNamespace(spatial_dimension="3D")

    attrs = BrukerCartesianRawRecoStrategy._base_attrs(handler)

    assert attrs["spatial_dimension"] == "3D"
    assert not REMOVED_DUPLICATE_ATTRS & set(attrs)


def test_spectroscopy_result_names_the_cut_off_once(monkeypatch):
    strategy = BrukerSpectroRecoStrategy()
    _stub_spectroscopy_reco(strategy, monkeypatch)

    result = strategy.reconstruct(
        _small_spectroscopy_handler(), data_precision="float32"
    )

    assert result.attrs["cut_off"] == 2
    assert not REMOVED_DUPLICATE_ATTRS & set(result.attrs)


@pytest.mark.parametrize("db", [True, False])
def test_raw_reco_raises_when_the_fov_shift_fails_for_every_db_setting(db):
    """Regression test: ``db=False`` used to continue unshifted.

    The synthetic scan has two slice packages of 1 and 5 slices but 2
    slices of data, so ``shift_image`` cannot assign the packages.
    """
    handler = _synthetic_raw_handler()
    handler.slices_per_pack = np.array([1, 5])

    with pytest.raises(ValueError, match="shift_image") as raised:
        BrukerRawRecoStrategy().reco_rawdata(
            handler=handler,
            values=handler.fid_pp,
            postprocess_fid=False,
            reshape_fid=False,
            shift_xarray=True,
            db=db,
            data_precision=np.float32,
        )

    assert raised.value.__cause__ is not None
    assert "slice packages" in str(raised.value)
    assert "shift_xarray=False" in str(raised.value)
    assert ("Full traceback" in str(raised.value)) is db


def test_raw_reco_without_the_fov_shift_ignores_a_failing_shift_setup():
    handler = _synthetic_raw_handler()
    handler.slices_per_pack = np.array([1, 5])

    reco = BrukerRawRecoStrategy().reco_rawdata(
        handler=handler,
        values=handler.fid_pp,
        postprocess_fid=False,
        reshape_fid=False,
        shift_xarray=False,
        db=False,
        data_precision=np.float32,
    )

    assert reco.attrs["domain"] == "spatial"


@pytest.mark.parametrize(
    "strategy_class", [BrukerRawRecoStrategy, BrukerMGERawRecoStrategy]
)
def test_reco_rawdata_without_spatial_reconstruction_returns_kspace(strategy_class):
    kspace = xr.DataArray(
        np.ones((2, 3), dtype=np.complex64),
        dims=("phase_idx", "read_idx"),
    )
    handler = SimpleNamespace(expnum=1, fid=kspace.data)

    reco = strategy_class().reco_rawdata(
        handler=handler,
        values=kspace.data,
        kspace_xarray=kspace,
        postprocess_fid=False,
        reshape_fid=False,
        set_kspace_xarray=False,
        shift_xarray=False,
        set_data=False,
        data_precision=np.float32,
    )

    assert reco is not kspace
    np.testing.assert_array_equal(reco, kspace)
    assert "domain" not in kspace.attrs
    assert reco.attrs["domain"] == "kspace"


@pytest.mark.parametrize(
    "strategy_class", [BrukerRawRecoStrategy, BrukerMGERawRecoStrategy]
)
def test_reco_rawdata_ignores_processed_arrays_left_on_the_handler(strategy_class):
    """Only explicit inputs or the acquired FID enter the reconstruction."""
    kspace = xr.DataArray(
        np.ones((2, 3), dtype=np.complex64),
        dims=("phase_idx", "read_idx"),
    )
    handler = SimpleNamespace(
        expnum=1,
        fid=kspace.data,
        fid_pp=np.full_like(kspace.data, 999.0),
        kspace_xarray=kspace * 999.0,
    )
    options = dict(
        handler=handler,
        postprocess_fid=False,
        reshape_fid=False,
        set_kspace_xarray=False,
        shift_xarray=False,
        set_data=False,
        data_precision=np.float32,
    )

    with pytest.raises(ValueError, match="requires an explicit kspace_xarray"):
        strategy_class().reco_rawdata(**options)
    np.testing.assert_array_equal(
        strategy_class()._fid_for_calculation(handler, False, np.float32),
        kspace.data,
    )


@pytest.mark.parametrize("db", [True, False])
def test_mge_raw_reco_raises_when_the_fov_shift_fails_for_every_db_setting(
    monkeypatch, db
):
    """Regression test the MGE strategy.

    The k-space array of the stand-in handler is set beforehand and
    ``shift_image`` is replaced by a function that fails, so that only the
    shift stage of the MGE strategy is exercised.
    """

    def failing_shift_image(**_kwargs):
        raise RuntimeError("synthetic shift failure")

    monkeypatch.setattr(raw_module, "shift_image", failing_shift_image)
    kspace = np.ones((2, 3), dtype=np.complex64)
    handler = SimpleNamespace(
        expnum=1,
        fid_pp=kspace,
        kspace=kspace,
        kspace_xarray=xr.DataArray(kspace, dims=("phase_idx", "read_idx")),
    )

    with pytest.raises(ValueError, match="shift_image") as raised:
        BrukerMGERawRecoStrategy().reco_rawdata(
            handler=handler,
            values=kspace,
            kspace_xarray=handler.kspace_xarray,
            postprocess_fid=False,
            reshape_fid=False,
            set_kspace_xarray=False,
            shift_xarray=True,
            db=db,
            data_precision=np.float32,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "synthetic shift failure" in str(raised.value)
    assert "shift_xarray=False" in str(raised.value)
    assert ("Full traceback" in str(raised.value)) is db


def test_agilent_and_magritek_share_fid_spectroscopy_helpers():
    assert issubclass(AgilentSpectroRecoStrategy, FidSpectroRecoStrategy)
    assert issubclass(MagritekSpectroRecoStrategy, FidSpectroRecoStrategy)
    assert not issubclass(MagritekSpectroRecoStrategy, AgilentSpectroRecoStrategy)
    assert (
        AgilentSpectroRecoStrategy._apply_zero_fill
        is FidSpectroRecoStrategy._apply_zero_fill
    )
    assert (
        MagritekSpectroRecoStrategy._apply_zero_fill
        is FidSpectroRecoStrategy._apply_zero_fill
    )


def test_fid_spectro_reconstruct_combines_channels_with_public_utility():
    class Handler(SimpleNamespace):
        def set_primary_data(self, data, name):
            self.primary_data = data
            self.primary_name = name

    fid = np.array([[[1, 2, 1, 0], [0, 1, 0, 1]]], dtype=np.complex64)
    handler = Handler(fid=fid, dwell_time_s=0.001)

    reco = FidSpectroRecoStrategy().reconstruct(
        handler,
        append_parameters=False,
        remove_dc=False,
    )

    assert "channel_idx" not in reco.dims
    assert reco.attrs["channel_combination"] == "rss"
    assert "coil_combination:rss" in reco.attrs["processing_log"]
    assert handler.primary_data is reco
    assert handler.primary_name == "data"


def _structural_reference_digest(array):
    """Hash samples, labelled coordinates and attributes without scanner data."""
    import hashlib
    import json

    def stable(value):
        if isinstance(value, dict):
            return {
                key: stable(value[key])
                for key in sorted(value)
                if key != "csi_sort_source_hash"
            }
        if isinstance(value, np.ndarray):
            return (str(value.dtype), value.shape, value.tolist())
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        return value

    payload = (
        stable(np.asarray(array.values)),
        array.dims,
        array.name,
        [
            (key, coord.dims, stable(coord.values), stable(coord.attrs))
            for key, coord in sorted(array.coords.items())
        ],
        stable(array.attrs),
    )

    def encode(value):
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Unsupported synthetic reference metadata: {type(value).__name__}"
        )

    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=encode
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _structural_reference_result(kind, spatial, precision, monkeypatch):
    """Reconstruct deterministic synthetic arrays through the public stages."""
    if kind in ("raw", "mge"):
        handler = _synthetic_raw_handler()
        strategy = BrukerRawRecoStrategy()
        if kind == "mge":
            strategy = BrukerMGERawRecoStrategy()
            handler.fid_pp = handler.fid_pp[:, :, :, :, :, 0, :]
            handler.fat_water_separation = "Off"
        else:
            handler.fid_pp = handler.fid_pp[:, :, :, :, 0, :, :]
            handler.fat_water_separation = "Off"
        handler.fid_pp = np.arange(handler.fid_pp.size).reshape(
            handler.fid_pp.shape
        ).astype(np.complex128) * (1 + 0.375j)
        handler.fid = handler.fid_pp.copy()
        if kind == "mge":
            handler.kspace_xarray = xr.DataArray(
                handler.fid_pp,
                dims=(
                    "acquisition_idx",
                    "repetition_idx",
                    "slice_idx",
                    "phase_idx",
                    "channel_idx",
                    "read_idx",
                ),
            )
        return strategy.reco_rawdata(
            handler,
            values=handler.fid,
            kspace_xarray=handler.kspace_xarray if kind == "mge" else None,
            postprocess_fid=False,
            reshape_fid=False,
            set_kspace_xarray=kind == "raw",
            shift_xarray=False,
            set_data=spatial,
            data_precision=precision,
        )
    if kind == "csi":
        handler = _linear_stock_csi_handler(cut_off=2 if spatial else 0)
        strategy = BrukerCsiRecoStrategy()
        sorted_data = strategy.sort_kspace(handler, data_type="fid")
        return strategy._sorted_csi_to_data(
            sorted_data, handler, data_precision=precision
        )
    handler = _small_spectroscopy_handler(cut_off=2 if spatial else 0)
    strategy = BrukerSpectroRecoStrategy()
    _stub_spectroscopy_reco(strategy, monkeypatch)
    return strategy.reconstruct(
        handler, data_precision=precision, append_parameters=False, shift_xarray=False
    )


# Filled from the unchanged reconstruction code before the reconstruction rewrite. Exact digests
# detect changed samples, dtype, dimensions, coordinates, units or provenance.
# The source-code hash changes with a structural edit and is checked separately.
# Synthetic baseline: working checkout on 6 October 2026, using
# NumPy 2.2.6, SciPy 1.18.1 and xarray 2025.6.1. Structural tolerance is zero.
# On 29 October 2026 the six raw digests were re-recorded after the fixture's
# patient_pos attribute became Foot_Supine; samples, coordinates and all other
# attributes were unchanged.
# The spatial raw and MGE digests changed on 22 November 2026: the
# read axis has 3 voxels, and odd axes now put the FOV centre at n / 2 as
# 2dseq does.  test_bruker_raw_transform.py checks this transform independently.
_STRUCTURAL_REFERENCES = {
    (
        "raw",
        False,
        "float16",
    ): "28f8a10e102ee56fef732ef32a595e0312369a35d10ea14fed4b4fa8339b1209",
    (
        "raw",
        False,
        "float32",
    ): "0010bf9a5561661eab6c4417f085dba53cc0d4c1bb62642382bcbd9c3d0e515d",
    (
        "raw",
        False,
        "float64",
    ): "0010bf9a5561661eab6c4417f085dba53cc0d4c1bb62642382bcbd9c3d0e515d",
    (
        "raw",
        True,
        "float16",
    ): "dfa5f9da8b78bf7102cd7d3e36e7404d1753d08cb115a42000072f92897835a0",
    (
        "raw",
        True,
        "float32",
    ): "dfa5f9da8b78bf7102cd7d3e36e7404d1753d08cb115a42000072f92897835a0",
    (
        "raw",
        True,
        "float64",
    ): "fb5444417972e587d190ac6c4110a2f1747c379b759f2f0c73cafeb9bc9187cb",
    (
        "mge",
        False,
        "float16",
    ): "405f0d3e4da98b3ed754a78af543c6038b39f7ca104c71cc56e155dad1ab27df",
    (
        "mge",
        False,
        "float32",
    ): "405f0d3e4da98b3ed754a78af543c6038b39f7ca104c71cc56e155dad1ab27df",
    (
        "mge",
        False,
        "float64",
    ): "405f0d3e4da98b3ed754a78af543c6038b39f7ca104c71cc56e155dad1ab27df",
    (
        "mge",
        True,
        "float16",
    ): "773a7cd31a0674a530a486ceae1e12cd1e08223a581eb5cb5ef5a1923ab96f73",
    (
        "mge",
        True,
        "float32",
    ): "773a7cd31a0674a530a486ceae1e12cd1e08223a581eb5cb5ef5a1923ab96f73",
    (
        "mge",
        True,
        "float64",
    ): "f0c8b174bc951200e8adb64e7d1b5c8f49901e8f536400eab347eb0a6b1f9880",
    (
        "csi",
        False,
        "float16",
    ): "2f392064ee4d345d8e3ad5bd6463f1cae97dfb31c4bb51ac53d41be7d6799ccf",
    (
        "csi",
        False,
        "float32",
    ): "2f392064ee4d345d8e3ad5bd6463f1cae97dfb31c4bb51ac53d41be7d6799ccf",
    (
        "csi",
        False,
        "float64",
    ): "e714990bdf8f48b08cdfbb9c122726b65dfa4bdba34fb9b33acc7d530bdbb6ac",
    (
        "csi",
        True,
        "float16",
    ): "85d83231236a6bb23e1ac68826c32ec550eb7e7b227de91480f62317169832ba",
    (
        "csi",
        True,
        "float32",
    ): "85d83231236a6bb23e1ac68826c32ec550eb7e7b227de91480f62317169832ba",
    (
        "csi",
        True,
        "float64",
    ): "541a49b2d6abc9ddd12f7984447abeb4764c4e684bda325cee7ac80802d73f60",
    (
        "spectroscopy",
        False,
        "float16",
    ): "09ea28255a821997572c9095671aea83274040e81f5a8a03581018dfc27289bf",
    (
        "spectroscopy",
        False,
        "float32",
    ): "09ea28255a821997572c9095671aea83274040e81f5a8a03581018dfc27289bf",
    (
        "spectroscopy",
        False,
        "float64",
    ): "09ea28255a821997572c9095671aea83274040e81f5a8a03581018dfc27289bf",
    (
        "spectroscopy",
        True,
        "float16",
    ): "f904c6704565b8f280722c382c647d8d26e41085980f0d26e816829f16e36805",
    (
        "spectroscopy",
        True,
        "float32",
    ): "f904c6704565b8f280722c382c647d8d26e41085980f0d26e816829f16e36805",
    (
        "spectroscopy",
        True,
        "float64",
    ): "f904c6704565b8f280722c382c647d8d26e41085980f0d26e816829f16e36805",
}


@pytest.mark.parametrize("kind", ["raw", "mge", "csi", "spectroscopy"])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("precision", ["float16", "float32", "float64"])
def test_structural_reconstruction_reference(kind, spatial, precision, monkeypatch):
    result = _structural_reference_result(kind, spatial, precision, monkeypatch)
    if kind == "spectroscopy" and precision == "float64":
        # Spectroscopy now stores float64 results; compare the former representation while
        # the independent float64 FFT regression checks retained precision.
        assert result.dtype == np.complex128
        result = result.copy(data=result.data.astype(np.complex64))
    if kind == "csi":
        result = result.copy(deep=True)
        extent = []
        for dim in ("read_idx", "phase_idx", "slice_idx"):
            extent.extend(result[dim].attrs["extent_mm"])
        for axis, dim in enumerate(("read_idx", "phase_idx", "slice_idx")):
            # A separate test verifies the corrected extent pairs independently; keep the
            # historical digest of all unaffected samples and metadata.
            result[dim].attrs["extent_mm"] = extent[axis]
        # The trace now also names the half-voxel ramp of odd lengths; the
        # samples are unchanged, so restore the former text for the digest.
        trace = json.loads(result.attrs["csi_reconstruction_trace"])
        assert trace["fft_shifts"] == (
            "ifftshift_before_fftshift_after_each_spatial_ifft;"
            "odd_length_half_voxel_phase_ramp_before_ifft"
        )
        trace["fft_shifts"] = "ifftshift_before_fftshift_after_each_spatial_ifft"
        result.attrs["csi_reconstruction_trace"] = json.dumps(trace, sort_keys=True)
    # The second names were removed of the cut-off, the spatial dimension and
    # the frequency axes; restoring them from the canonical values keeps the
    # historical digest of everything else.
    result = result.copy()
    result.attrs = dict(result.attrs)
    if "cut_off" in result.attrs and kind in ("spectroscopy", "csi"):
        result.attrs["applied_cut_off"] = result.attrs["cut_off"]
    if "spatial_dimension" in result.attrs:
        result.attrs["spat_dim"] = result.attrs["spatial_dimension"]
    for name in ("freq_axis_hz", "freq_axis_ppm"):
        # The axes now live only in the coordinates of the same name.
        if name in result.coords:
            assert name not in result.attrs
            result.attrs[name] = result.coords[name].values
    if "freq_axis_hz" in result.attrs:
        result.attrs["freq_range_hz"] = result.attrs["freq_axis_hz"]
        result.attrs["freq_range_Hz"] = result.attrs["freq_axis_hz"]
    if "freq_axis_ppm" in result.attrs:
        result.attrs["freq_range_ppm"] = result.attrs["freq_axis_ppm"]
    assert (
        _structural_reference_digest(result)
        == _STRUCTURAL_REFERENCES[(kind, spatial, precision)]
    )


def test_packaged_imaging_precision_declarations_preserve_decision_dec2():
    """Every declared imaging storage precision stays explicitly float32."""
    declarations = [
        options["data_precision"]
        for _strategy, options in RECONSTRUCTIONS.values()
        if "data_precision" in options
    ]
    assert len(declarations) == 14
    assert set(declarations) == {"float32"}


def test_rare_reconstruction_uses_current_fid_at_each_precision(monkeypatch):
    """A float64 RARE result must not reuse a rounded float16 k-space.

    Synthetic acquired samples include a 1e-8 component lost in complex64.
    The reference is an independent fresh float64 reconstruction. Both runs
    use the same deterministic Fourier operations, so exact equality is
    required; acquired samples must remain unchanged.
    """
    from hypermri.reconstruction.precision import convert_for_calculation

    def make_handler():
        handler = _synthetic_raw_handler()
        handler.fat_water_separation = "Off"
        handler.fid = handler.fid_pp[:, :, :, :, 0, :, :].astype(np.complex128)
        handler.fid += 1e-8 * (1 + 1j)
        del handler.fid_pp
        return handler

    strategy = BrukerRareRawRecoStrategy()
    monkeypatch.setattr(
        strategy,
        "postprocess_fid_rare",
        lambda handler, data_precision: convert_for_calculation(
            handler.fid, data_precision
        ),
    )
    handler = make_handler()
    acquired = handler.fid.copy()
    options = {"append_parameters": False, "shift_xarray": False}
    strategy.reconstruct(handler, data_precision="float16", **options)
    repeated = strategy.reconstruct(handler, data_precision="float64", **options)
    fresh = strategy.reconstruct(make_handler(), data_precision="float64", **options)

    xr.testing.assert_identical(repeated, fresh)
    np.testing.assert_array_equal(handler.fid, acquired)


def test_spectroscopy_float64_preserves_the_requested_storage_precision(monkeypatch):
    """The final spectral FFT retains float64 acquisition precision.

    The synthetic reference is NumPy's FFT of the unrounded acquired FID.
    Relative tolerance 1e-14 allows float64 transform rounding and would
    detect the previous complex64 quantization of the temporal signal.
    """
    strategy = BrukerSpectroRecoStrategy()
    handler = _small_spectroscopy_handler(cut_off=0)
    handler.fid = handler.fid.astype(np.complex128) + (1e-8 + 2e-8j)
    handler.spatial_dimension = "2D"
    acquired = handler.fid.copy()
    monkeypatch.setattr(
        strategy, "set_data", lambda kspace_xarray, **kwargs: kspace_xarray.copy()
    )
    result = strategy.reconstruct(
        handler, data_precision="float64", append_parameters=False, shift_xarray=False
    )
    assert result.dtype == np.complex128
    np.testing.assert_allclose(
        result.values.squeeze(),
        np.fft.fftshift(np.fft.fft(acquired)),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(handler.fid, acquired)


def test_spectroscopy_records_selected_reco_parameters_as_a_snapshot(monkeypatch):
    strategy = BrukerSpectroRecoStrategy()
    handler = _small_spectroscopy_handler(cut_off=0)
    handler.spatial_dimension = "2D"
    handler.reco_parameter_source = "pdata/1/reco"
    handler.reco = {"RecoObjectsPerRepetition": 1, "RECO_time": "synthetic first"}
    monkeypatch.setattr(
        strategy, "set_data", lambda kspace_xarray, **kwargs: kspace_xarray.copy()
    )
    first = strategy.reconstruct(handler, append_parameters=False, shift_xarray=False)
    handler.reco_parameter_source = "pdata/2/reco"
    handler.reco = {"RecoObjectsPerRepetition": 2, "RECO_time": "synthetic second"}
    second = strategy.reconstruct(handler, append_parameters=False, shift_xarray=False)
    assert first.attrs["bruker_reco_parameter_source"] == "pdata/1/reco"
    assert second.attrs["bruker_reco_parameter_source"] == "pdata/2/reco"
    assert first.attrs["bruker_reco_objects_per_repetition"] == 1
    assert second.attrs["bruker_reco_objects_per_repetition"] == 2
    assert first.attrs["bruker_reco_time"] == "synthetic first"
    assert second.attrs["bruker_reco_time"] == "synthetic second"
    assert (first.sizes["phase_idx"], second.sizes["phase_idx"]) == (1, 2)


@pytest.mark.parametrize("append_parameters", [False, True])
def test_spectroscopy_keeps_patient_position_without_changing_spectrum(
    monkeypatch, append_parameters
):
    """STEAM uses this shared spectroscopy path and must retain position."""
    strategy = BrukerSpectroRecoStrategy()
    handler = _small_spectroscopy_handler(cut_off=0)
    handler.patient_pos = "Head_Prone"
    handler.spatial_dimension = "2D"
    acquired = handler.fid.copy()
    monkeypatch.setattr(
        strategy, "set_data", lambda kspace_xarray, **kwargs: kspace_xarray.copy()
    )

    result = strategy.reconstruct(
        handler,
        data_precision="float64",
        append_parameters=append_parameters,
        shift_xarray=False,
    )

    assert result.attrs["patient_pos"] == "Head_Prone"
    np.testing.assert_allclose(
        result.values.squeeze(),
        np.fft.fftshift(np.fft.fft(acquired.astype(np.complex128))),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(handler.fid, acquired)


def test_raw_preprocessing_failure_reaches_the_caller(monkeypatch):
    """An invalid preprocessing stage must not reconstruct the acquired layout."""
    handler = _synthetic_raw_handler()
    handler.fid = handler.fid_pp
    strategy = BrukerRawRecoStrategy()
    failure = TypeError("synthetic preprocessing defect")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(strategy, "postprocess_fid_file", fail)
    with pytest.raises((TypeError, ValueError), match="synthetic preprocessing defect"):
        strategy.reco_rawdata(
            handler, reshape_fid=False, shift_xarray=False, set_data=False
        )


def test_seq2d_reconstruction_failure_reaches_the_caller(monkeypatch):
    """A failed scanner reconstruction differs from an absent 2dseq file."""
    strategy = Brukerseq2dRecoStrategy()
    failure = RuntimeError("synthetic scanner defect")

    def fail(handler, **kwargs):
        raise failure

    monkeypatch.setattr(strategy, "_reconstruct_internal", fail)
    with pytest.raises(RuntimeError, match="synthetic scanner defect") as caught:
        strategy.reconstruct(SimpleNamespace(expnum=1))
    assert caught.value is failure


def test_raw_reconstruct_keeps_handler_intermediates_and_metadata_unchanged():
    """A stale legacy intermediate must not affect the public raw route."""
    handler = _synthetic_raw_handler()
    handler.fat_water_separation = "Off"
    handler.fid = handler.fid_pp[:, :, :, :, 0, :, :].astype(np.complex128)
    handler.fid_pp = np.full_like(handler.fid, 999.0)
    legacy = handler.fid_pp.copy()
    acquired = handler.fid.copy()
    strategy = BrukerRawRecoStrategy()
    result = strategy.reconstruct(
        handler, postprocess_fid=False, reshape_fid=False, shift_xarray=False
    )
    assert not hasattr(handler, "kspace")
    assert not hasattr(handler, "kspace_xarray")
    assert not hasattr(handler, "data")
    np.testing.assert_array_equal(handler.fid, acquired)
    np.testing.assert_array_equal(handler.fid_pp, legacy)
    assert np.max(np.abs(result)) < 999.0


@pytest.mark.parametrize("with_noise", [False, True])
def test_csi_phase_failure_never_returns_unshifted_signal(monkeypatch, with_noise):
    """Spatial signal and covariance require the same successful phase operator."""
    from hypermri.reconstruction.strategies.bruker import csi

    def fail(array):
        raise RuntimeError("synthetic CSI phase defect")

    data = xr.DataArray(np.ones((2, 2), dtype=complex), dims=("read_idx", "phase_idx"))
    step = csi._csi_fov_phase_weights if with_noise else csi._shift_csi_fov
    with pytest.raises(RuntimeError, match="synthetic CSI phase defect"):
        step(data, fail)


def test_csi_source_hash_describes_the_current_sorting_implementation():
    import hashlib
    from pathlib import Path

    import hypermri.reconstruction.csi_raw as raw_module

    directory = Path(raw_module.__file__).parent
    expected = hashlib.sha256(
        Path(raw_module.__file__).read_bytes()
        + (directory / "strategies/bruker/csi.py").read_bytes()
    ).hexdigest()
    result = BrukerCsiRecoStrategy().sort_kspace(
        _linear_stock_csi_handler(), data_type="fid"
    )
    assert result.attrs["csi_sort_source_hash"] == expected
