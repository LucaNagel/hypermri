"""Opt-in scanner phase-ramp contract, using synthetic complex Gaussian samples.

A comparison with scanner images found the slices half a slice beside the plain forward slice
transform.  The slice-encoding transform applies that half-slice
translation itself (positive Nyquist branch), so scanner mode gives the
independent image and only adds metadata checks and provenance. A 1e-6 peak-amplitude tolerance covers complex64
storage; complex128 uses 1e-12. No acquired data or scanner paths are included.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.strategies.bruker import raw
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)

_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _handler():
    return SimpleNamespace(
        expnum=1,
        type="<Bruker:FLASH>",
        fid=np.array([1 + 2j]),
        spatial_dimension="3D",
        patient_pos="Head_Supine",
        slice_orientation=np.array(["coronal"]),
        read_orientation=np.array(["H_F"]),
        matrix=np.array([4, 6, 8]),
        enc_matrix=np.array([4, 6, 8]),
        resolution_mm=np.array([1.0, 1.0, 1.0]),
        read_offset_mm=0.0,
        phase_offset_mm=0.0,
        slice_offset_mm=-3.0,
        n_slice_packs=1,
        slices_per_pack=(1,),
        method={
            "PVM_NSPacks": 1,
            "PVM_SPackArrNSlices": np.array([1]),
            "PVM_SPackArrReadOffset": np.array([0.0]),
            "PVM_SPackArrPhase1Offset": np.array([0.0]),
            "PVM_SPackArrSliceOffset": np.array([-3.0]),
        },
        reco={"RECO_rotate": np.array([[0.5], [0.5], [0.8125]])},
        reco_parameter_source="pdata/1/reco",
        n_inversion_time=1,
        git_commit_hash=None,
    )


@pytest.fixture
def synthetic_reconstruction(monkeypatch):
    """Isolate phase shifting and the real Fourier stages from file decoding."""
    rng = np.random.default_rng(36)
    values = rng.normal(size=(4, 6, 8, 2)) + 1j * rng.normal(size=(4, 6, 8, 2))
    kspace = xr.DataArray(
        values,
        dims=(*_SPATIAL_DIMS, "repetition_idx"),
        coords={
            dim: (dim, np.arange(size), {"unit": "mm"})
            for dim, size in zip(_SPATIAL_DIMS, values.shape, strict=False)
        },
        attrs={
            "spatial_dimension": "3D",
            "domain": "kspace",
            "provenance": "synthetic",
        },
    )
    kspace = kspace.expand_dims({"acquisition_idx": [0], "channel_idx": [0]}).transpose(
        "acquisition_idx", *_SPATIAL_DIMS, "repetition_idx", "channel_idx"
    )
    monkeypatch.setattr(raw, "_attach_image_geometry", lambda data, handler: data)
    strategy = BrukerRawRecoStrategy()
    monkeypatch.setattr(strategy, "_label_raw_kspace", lambda handler, data: kspace)
    return strategy, _handler(), kspace


def _reconstruct(strategy, handler, **options):
    return strategy.reconstruct(
        handler, postprocess_fid=False, reshape_fid=False, **options
    )


def _translate_slice_values(values, axis):
    """Translate by -0.5 slice with the positive Nyquist branch."""
    count = values.shape[axis]
    frequencies = np.fft.fftfreq(count)
    frequencies[count // 2] = 0.5
    shape = [1] * values.ndim
    shape[axis] = count
    phase = np.exp(-2j * np.pi * frequencies * -0.5).reshape(shape)
    return np.fft.ifft(np.fft.fft(values, axis=axis) * phase, axis=axis)


def test_default_keeps_the_acquisition_phase_ramp(synthetic_reconstruction):
    """The default applies the acquisition ramp and the half-slice centring."""
    strategy, handler, kspace = synthetic_reconstruction
    image = _reconstruct(strategy, handler, data_precision="float64")
    centred_indices = np.arange(8) - 4
    shifted = (
        kspace.values
        * np.exp(2j * np.pi * centred_indices * -3.0 / 8)[
            None, None, None, :, None, None
        ]
    )
    expected = _translate_slice_values(
        np.fft.fftshift(np.fft.fft(np.fft.fftshift(shifted, axes=3), axis=3), axes=3),
        axis=3,
    )
    for axis in (1, 2):
        expected = np.fft.ifftshift(
            np.fft.ifft(
                np.fft.fftshift(expected, axes=axis), axis=axis, norm="forward"
            ),
            axes=axis,
        )
    np.testing.assert_allclose(
        image, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    assert "fov_shift_source" not in image.attrs


@pytest.mark.parametrize("precision,tolerance", [("float32", 1e-6), ("float64", 1e-12)])
@pytest.mark.parametrize("permuted", [False, True])
def test_scanner_mode_gives_the_independent_image(
    synthetic_reconstruction, monkeypatch, precision, tolerance, permuted
):
    strategy, handler, kspace = synthetic_reconstruction
    if permuted:
        kspace = kspace.transpose(
            "channel_idx",
            "slice_idx",
            "repetition_idx",
            "phase_idx",
            "read_idx",
            "acquisition_idx",
        )
        monkeypatch.setattr(strategy, "_label_raw_kspace", lambda handler, data: kspace)
    original = kspace.copy(deep=True)
    acquired = handler.fid.copy()
    independent = _reconstruct(strategy, handler, data_precision=precision)
    image = _reconstruct(
        strategy, handler, data_precision=precision, fov_shift_source="scanner"
    )
    expected = independent.values
    np.testing.assert_allclose(
        image, expected, rtol=0, atol=tolerance * np.abs(expected).max()
    )
    assert image.dtype == independent.dtype
    assert image.dims == independent.dims
    xr.testing.assert_identical(
        image.coords.to_dataset(), independent.coords.to_dataset()
    )
    assert image.attrs["num_image_fov_shift"] == 1
    assert image.attrs["fov_shift_source"] == "scanner"
    assert image.attrs["fov_shift_vox"] == (0.0, 0.0, -3.0)
    assert image.attrs["bruker_reco_parameter_source"] == "pdata/1/reco"
    assert image.attrs["bruker_reco_rotate"] == (0.5, 0.5, 0.8125)
    xr.testing.assert_identical(
        image,
        _reconstruct(
            strategy, handler, data_precision=precision, fov_shift_source="scanner"
        ),
    )
    xr.testing.assert_identical(kspace, original)
    np.testing.assert_array_equal(handler.fid, acquired)
    xr.testing.assert_identical(
        independent,
        _reconstruct(
            strategy, handler, data_precision=precision, fov_shift_source="acquisition"
        ),
    )


@pytest.mark.parametrize("options", [{"shift_xarray": False}, {"fov_shift_sign": -1.0}])
def test_low_level_explicit_vector_rejects_conflicting_options(options):
    with pytest.raises(ValueError, match="shift_xarray=True and fov_shift_sign=1"):
        BrukerRawRecoStrategy().reco_rawdata(
            _handler(), fov_shift_vox=(0, 0, -3.5), **options
        )


def test_scanner_mode_rejects_data_that_disagree_with_declared_matrix(
    synthetic_reconstruction, monkeypatch
):
    strategy, handler, kspace = synthetic_reconstruction
    incomplete = kspace.isel(read_idx=slice(1, None))
    monkeypatch.setattr(strategy, "_label_raw_kspace", lambda handler, data: incomplete)
    with pytest.raises(ValueError, match="full acquisition matrix"):
        _reconstruct(strategy, handler, fov_shift_source="scanner")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("spatial_dimension", "2D", "3D"),
        ("type", "<Bruker:RARE>", "FLASH"),
        ("patient_pos", "Head_Prone", "Head_Supine"),
        ("slice_orientation", "axial", "coronal"),
        ("read_orientation", "L_R", "H_F"),
        ("n_slice_packs", 2, "PVM_NSPacks"),
        ("slices_per_pack", (4, 4), "PVM_SPackArrNSlices"),
        ("matrix", [4, 6, 7], "matrix"),
        ("matrix", [4, 6, np.nan], "matrix"),
        ("enc_matrix", [4, 6, 6], "PVM_EncMatrix"),
        ("resolution_mm", [1, 1, 0], "resolution_mm"),
        ("reco_parameter_source", None, "reco_parameter_source"),
        ("reco_parameter_source", "pdata/2/reco", "reco_parameter_source"),
        ("read_offset_mm", 1.0, "ReadOffset"),
        ("phase_offset_mm", 1.0, "Phase1Offset"),
        ("slice_offset_mm", -2.0, "SliceOffset"),
        ("method", None, "method"),
        ("reco", None, "RECO_rotate"),
        ("reco", {"RECO_rotate": [0.5, 0.5, 0.8125]}, "RECO_rotate"),
        ("reco", {"RECO_rotate": [[0.5], [0.5], [np.nan]]}, "RECO_rotate"),
        ("reco", {"RECO_rotate": [[0.6], [0.5], [0.8125]]}, "in-plane"),
        ("reco", {"RECO_rotate": [[0.5], [0.5], [0.7]]}, "centering"),
    ],
)
def test_unsupported_scanner_metadata_fails_before_fid_read(
    monkeypatch, field, value, match
):
    handler = _handler()
    setattr(handler, field, value)
    handler.fid = None
    monkeypatch.setattr(
        BrukerRawRecoStrategy,
        "_load_fid_on_demand",
        lambda self, h: pytest.fail("FID read before validation"),
    )
    with pytest.raises(ValueError, match=match):
        BrukerRawRecoStrategy().reconstruct(handler, fov_shift_source="scanner")


@pytest.mark.parametrize(
    "key",
    [
        "PVM_NSPacks",
        "PVM_SPackArrNSlices",
        "PVM_SPackArrReadOffset",
        "PVM_SPackArrPhase1Offset",
        "PVM_SPackArrSliceOffset",
    ],
)
def test_missing_acquisition_metadata_is_not_defaulted(monkeypatch, key):
    handler = _handler()
    del handler.method[key]
    monkeypatch.setattr(
        BrukerRawRecoStrategy,
        "_load_fid_on_demand",
        lambda self, h: pytest.fail("FID read before validation"),
    )
    with pytest.raises(ValueError, match=key):
        BrukerRawRecoStrategy().reconstruct(handler, fov_shift_source="scanner")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"shift_xarray": False}, "shift_xarray"),
        ({"fov_shift_sign": -1}, "fov_shift_sign"),
        ({"fov_shift_source": "guess"}, "fov_shift_source"),
    ],
)
def test_conflicting_options_fail_before_fid_read(monkeypatch, options, match):
    monkeypatch.setattr(
        BrukerRawRecoStrategy,
        "_load_fid_on_demand",
        lambda self, h: pytest.fail("FID read before validation"),
    )
    with pytest.raises(ValueError, match=match):
        BrukerRawRecoStrategy().reconstruct(
            _handler(), **{"fov_shift_source": "scanner", **options}
        )


def test_inherited_mge_strategy_rejects_scanner_mode_before_fid_read(monkeypatch):
    monkeypatch.setattr(
        BrukerRawRecoStrategy,
        "_load_fid_on_demand",
        lambda self, h: pytest.fail("FID read before validation"),
    )
    with pytest.raises(ValueError, match="FLASH"):
        BrukerMGERawRecoStrategy().reconstruct(_handler(), fov_shift_source="scanner")


@pytest.mark.parametrize(
    "key,attribute",
    [
        ("PVM_SPackArrReadOffset", "read_offset_mm"),
        ("PVM_SPackArrPhase1Offset", "phase_offset_mm"),
    ],
)
def test_nonzero_inplane_offsets_are_not_hidden_by_imaging_shift_defaults(
    monkeypatch, key, attribute
):
    handler = _handler()
    handler.method[key] = 1.0
    setattr(handler, attribute, 1.0)
    monkeypatch.setattr(
        BrukerRawRecoStrategy,
        "_load_fid_on_demand",
        lambda self, h: pytest.fail("FID read before validation"),
    )
    with pytest.raises(ValueError, match=key):
        BrukerRawRecoStrategy().reconstruct(handler, fov_shift_source="scanner")


@pytest.mark.parametrize(
    "strategy_name", ["BrukerRareRawRecoStrategy", "BrukerMGERawRecoStrategy"]
)
@pytest.mark.parametrize("source", ["acquisition", "scanner"])
def test_public_dispatch_rejects_options_other_strategies_would_ignore(
    monkeypatch, strategy_name, source
):
    from hypermri.exceptions import ReconstructionError
    from hypermri.loading import experiment as experiment_module
    from hypermri.reconstruction.strategies.bruker.rare import BrukerRareRawRecoStrategy

    strategy_type = {
        "BrukerRareRawRecoStrategy": BrukerRareRawRecoStrategy,
        "BrukerMGERawRecoStrategy": BrukerMGERawRecoStrategy,
    }[strategy_name]
    monkeypatch.setattr(
        strategy_type,
        "reconstruct",
        lambda *args, **kwargs: pytest.fail("Unsupported strategy was invoked"),
    )
    monkeypatch.setattr(
        "hypermri.reconstruction.selection.find_reconstruction",
        lambda *args: (strategy_type, {}),
    )
    experiment = experiment_module.MRIExperiment.__new__(
        experiment_module.MRIExperiment
    )
    experiment.handler = SimpleNamespace(id="synthetic", type="FLASH", vendor="bruker")
    experiment._forced_vendor = None
    with pytest.raises(ReconstructionError, match="fov_shift_source"):
        experiment.reconstruct("raw", fov_shift_source=source)
    assert experiment._reconstruction_status["raw"] == "Failed"


@pytest.mark.parametrize(
    "shift", [(0, 0), (0, 0, np.nan), (0, 0, np.inf), ((0, 0, -3.5),)]
)
def test_low_level_explicit_vector_rejects_invalid_values_before_preprocessing(
    monkeypatch, shift
):
    strategy = BrukerRawRecoStrategy()
    monkeypatch.setattr(
        strategy,
        "_fid_for_calculation",
        lambda *args: pytest.fail("Samples processed before validation"),
    )
    with pytest.raises(ValueError, match="fov_shift_vox"):
        strategy.reco_rawdata(_handler(), fov_shift_vox=shift)


def test_low_level_explicit_shift_rejects_already_shifted_input(
    synthetic_reconstruction,
):
    strategy, handler, kspace = synthetic_reconstruction
    kspace.attrs["num_image_fov_shift"] = 1
    original = kspace.copy(deep=True)
    with pytest.raises(ValueError, match="unshifted labelled 3D"):
        strategy.reco_rawdata(
            handler,
            postprocess_fid=False,
            reshape_fid=False,
            values=handler.fid,
            set_kspace_xarray=False,
            kspace_xarray=kspace,
            fov_shift_vox=(0, 0, -3.5),
        )
    xr.testing.assert_identical(kspace, original)
