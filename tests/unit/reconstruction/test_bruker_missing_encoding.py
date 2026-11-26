"""Reject missing Cartesian encoding before processing a custom B1 FID.

Fixtures contain invented samples and metadata only. Scalar ACQ_size exercises
the preceding padding failure; disabling padding exposes the original None
indexing in the k-space reshape. Neither path may guess an encoding matrix.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri import load_scan
from hypermri.exceptions import ReconstructionError
from hypermri.reconstruction.strategies.bruker.raw import (
    BrukerMGERawRecoStrategy,
    BrukerRawRecoStrategy,
)
from hypermri.sequences.B1Calibration import B1Calibration


def _missing_encoding_handler():
    return SimpleNamespace(
        expnum=1,
        fid=np.arange(128, dtype=np.float64).astype(np.complex128),
        enc_matrix=None,
        spatial_dimension=None,
        n_repetitions=1,
        n_receivers=1,
        method={"PVM_EncNReceivers": 1},
        acqp={
            "ACQ_scan_name": "<synthetic>",
            "ACQ_grad_matrix": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            "ACQ_experiment_mode": "SingleExperiment",
            "ACQ_size": 8,
            "NI": 1,
            "NR": 1,
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            "GO_block_size": "Standard_KBlock_Format",
        },
    )


@pytest.mark.parametrize("postprocess_fid", [True, False])
@pytest.mark.parametrize("db", [True, False])
@pytest.mark.parametrize("attribute_absent", [True, False])
@pytest.mark.parametrize(
    "strategy_class", [BrukerRawRecoStrategy, BrukerMGERawRecoStrategy]
)
def test_missing_encoding_is_rejected_before_fid_processing(
    strategy_class, postprocess_fid, db, attribute_absent, monkeypatch
):
    handler = _missing_encoding_handler()
    if attribute_absent:
        del handler.enc_matrix
    acquired = handler.fid.copy()
    strategy = strategy_class()
    calls = []
    original = strategy.postprocess_fid_file

    def record_processing(**kwargs):
        calls.append("postprocess_fid_file")
        return original(**kwargs)

    monkeypatch.setattr(strategy, "postprocess_fid_file", record_processing)

    with pytest.raises(ValueError, match="PVM_EncMatrix") as caught:
        strategy.reconstruct(handler, postprocess_fid=postprocess_fid, db=db)

    assert "method" in str(caught.value)
    assert "scanner_data" in str(caught.value)
    assert "seq2d" in str(caught.value)
    assert calls == []
    np.testing.assert_array_equal(handler.fid, acquired)
    assert handler.fid.dtype == np.complex128
    assert not {"fid_pp", "kspace", "kspace_xarray", "data"} & vars(handler).keys()


def test_explicit_labelled_kspace_does_not_require_an_encoding_matrix():
    handler = _missing_encoding_handler()
    kspace = xr.DataArray(
        np.arange(4, dtype=np.complex128),
        dims=("acquisition_idx",),
        coords={"acquisition_idx": ("acquisition_idx", [0, 1, 2, 3], {"unit": "s"})},
        attrs={"unit": "a.u."},
    )
    original = kspace.copy(deep=True)

    result = BrukerRawRecoStrategy().reco_rawdata(
        handler,
        postprocess_fid=False,
        reshape_fid=False,
        set_kspace_xarray=False,
        shift_xarray=False,
        set_data=False,
        kspace_xarray=kspace,
    )

    xr.testing.assert_identical(kspace, original)
    xr.testing.assert_identical(result, original.assign_attrs(domain="kspace"))


def test_b1_data_propagates_missing_encoding_and_keeps_explicit_scanner_data(
    synthetic_flash_scan, write_paravision_parameters, monkeypatch
):
    handler_input = _missing_encoding_handler()
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {"Method": "b1calibration", **handler_input.method},
    )
    write_paravision_parameters(synthetic_flash_scan / "acqp", handler_input.acqp)
    (synthetic_flash_scan / "fid").write_bytes(np.arange(256, dtype="<i4").tobytes())
    scan = load_scan(synthetic_flash_scan, mode="minimal", log_mode="critical")
    assert isinstance(scan, B1Calibration)
    assert scan.handler.enc_matrix is None
    scanner = scan.scanner_data.copy(deep=True)
    calls = []
    original = scan.reconstruct

    def record_reconstruction(reco_type, **options):
        calls.append(reco_type)
        return original(reco_type, **options)

    monkeypatch.setattr(scan, "reconstruct", record_reconstruction)

    with pytest.raises(ReconstructionError, match="PVM_EncMatrix") as direct:
        scan.reconstruct("raw", db=False)
    assert isinstance(direct.value.__cause__, ValueError)
    acquired = scan.handler.fid.copy()
    errors = []
    for _ in range(2):
        with pytest.raises(ReconstructionError, match="PVM_EncMatrix") as caught:
            _ = scan.primary_data
        errors.append(caught.value)

    assert errors[0] is errors[1]
    assert calls == ["raw", "raw"]
    assert "raw" not in scan.reconstructions
    assert "scanner_data" in str(errors[0])
    xr.testing.assert_identical(scan.scanner_data, scanner)
    np.testing.assert_array_equal(scan.handler.fid, acquired)
    assert not {"fid_pp", "kspace", "kspace_xarray", "data"} & vars(scan.handler).keys()
