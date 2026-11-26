"""Loading errors surface with their cause instead of a generic experiment.

All scan folders are synthetic and written into ``tmp_path``; they hold only
the parameter files that each case needs.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

from hypermri import load_scan
from hypermri.exceptions import DataLoadingError
from hypermri.vendors import get_vendor_handler
from hypermri.vendors.bruker import BrukerHandler

MODES = ["lazy", "minimal", "full"]


def _write_incomplete_scan(scan_dir: Path, write_parameters) -> None:
    """Write a ParaVision folder whose acqp lacks the scan name."""
    write_parameters(scan_dir / "method", {"Method": "<Bruker:FLASH>"})
    write_parameters(scan_dir / "acqp", {"NR": 1})


def _write_scan_with_unreadable_2dseq(scan_dir: Path, write_parameters) -> None:
    """Write a scan whose parameters load but whose 2dseq cannot be labelled.

    ``RECO_transposition`` with two different nonzero values is rejected by
    :meth:`BrukerHandler.undo_reco_transposition` when the data are read.
    """
    write_parameters(scan_dir / "method", {"Method": "<Bruker:FLASH>"})
    write_parameters(
        scan_dir / "acqp",
        {"ACQ_scan_name": "<synthetic>", "ACQ_patient_pos": "Head_Supine"},
    )
    visu_pars = {
        "VisuCoreSize": [2, 2],
        "VisuCreatorVersion": "7.0.0",
        "VisuCoreWordType": "_16BIT_SGN_INT",
        "VisuCoreByteOrder": "littleEndian",
        "VisuCoreFrameCount": 2,
        "VisuCoreDimDesc": ["spatial", "spatial"],
        "VisuFGOrderDescDim": 1,
        "VisuFGOrderDesc": ["(2, <FG_SLICE>, <>, 0, 0)"],
        "VisuCoreDataSlope": [1, 1],
        "VisuCoreDataOffs": [0, 0],
        "VisuCoreFrameType": "MAGNITUDE_IMAGE",
    }
    pdata = scan_dir / "pdata" / "1"
    write_parameters(scan_dir / "visu_pars", visu_pars)
    write_parameters(pdata / "visu_pars", visu_pars)
    write_parameters(pdata / "reco", {"RECO_transposition": [1, 2]})
    (pdata / "2dseq").write_bytes(np.zeros(8, dtype="<i2").tobytes())


@pytest.mark.parametrize("mode", MODES)
def test_load_scan_names_a_missing_path(tmp_path, mode):
    missing = tmp_path / "missing"

    with pytest.raises(DataLoadingError, match="the path does not exist") as info:
        load_scan(missing, mode=mode)

    assert info.value.path == str(missing)


def test_load_scan_names_a_folder_that_no_vendor_recognizes(tmp_path):
    with pytest.raises(DataLoadingError, match="no registered vendor recognizes"):
        load_scan(tmp_path)


def test_load_scan_names_an_unregistered_forced_vendor(tmp_path):
    with pytest.raises(
        DataLoadingError, match="force_vendor='nonsense' names no registered vendor"
    ):
        load_scan(tmp_path, force_vendor="nonsense")


@pytest.mark.parametrize("mode", MODES)
def test_incomplete_bruker_parameters_raise_with_the_scan_path(
    tmp_path, write_paravision_parameters, caplog, mode
):
    """Previously a half-constructed handler with ``type None`` was returned."""
    scan_dir = tmp_path / "scan"
    _write_incomplete_scan(scan_dir, write_paravision_parameters)

    with caplog.at_level(logging.ERROR), warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DataLoadingError, match="parameter files") as info:
            load_scan(scan_dir, mode=mode)

    assert str(scan_dir) in str(info.value)
    assert info.value.vendor == "bruker"
    assert isinstance(info.value.__cause__, KeyError)
    assert caplog.records == []


def test_full_load_names_an_unreadable_payload(tmp_path, write_paravision_parameters):
    scan_dir = tmp_path / "scan"
    _write_scan_with_unreadable_2dseq(scan_dir, write_paravision_parameters)
    BrukerHandler(scan_dir)

    with pytest.raises(DataLoadingError, match="Reading the data") as info:
        BrukerHandler(scan_dir, load_data=True)

    assert isinstance(info.value.__cause__, ValueError)
    assert "RECO_transposition" in str(info.value.__cause__)


@pytest.mark.parametrize("empty_file", [False, True], ids=["missing", "empty"])
def test_full_load_keeps_a_scan_without_seq2d(
    tmp_path, write_paravision_parameters, caplog, empty_file
):
    """Optional missing images give a precise warning and keep the scan."""
    from hypermri import Study

    scan_dir = tmp_path / "5"
    write_paravision_parameters(scan_dir / "method", {"Method": "<Bruker:FLASH>"})
    write_paravision_parameters(
        scan_dir / "acqp",
        {"ACQ_scan_name": "<synthetic>", "ACQ_patient_pos": "Head_Supine"},
    )
    pdata = scan_dir / "pdata" / "1"
    write_paravision_parameters(pdata / "reco", {"RECO_transposition": [1]})
    if empty_file:
        (pdata / "2dseq").touch()

    with caplog.at_level(logging.WARNING):
        scan = load_scan(scan_dir, mode="full", log_mode="warning")
        study = Study(
            tmp_path, load_data="full", log_mode="warning", verbose=False, sort=None
        )

    assert scan.handler.seq2d.size == 0
    assert scan.reconstructions == {}
    assert study[5].handler.seq2d.size == 0
    assert len(study) == 1
    records = [
        record for record in caplog.records if record.name.startswith("hypermri")
    ]
    assert not any(record.levelno >= logging.ERROR for record in records)
    assert any(
        "No seq2d data available for scan 5" in record.getMessage()
        and str(pdata / "2dseq") in record.getMessage()
        for record in records
    )
    assert not any("VisuCoreDimDesc" in record.getMessage() for record in records)


def test_get_vendor_handler_propagates_errors_of_a_recognized_scan(
    tmp_path, write_paravision_parameters
):
    scan_dir = tmp_path / "scan"
    _write_incomplete_scan(scan_dir, write_paravision_parameters)

    with pytest.raises(DataLoadingError):
        get_vendor_handler(scan_dir)
    assert get_vendor_handler(tmp_path / "missing") is None


def test_study_skips_an_unreadable_scan_and_logs_its_cause(
    tmp_path, write_paravision_parameters, caplog
):
    """Previously the half-constructed scan made the Study raise a KeyError."""
    from hypermri import Study

    _write_incomplete_scan(tmp_path / "1", write_paravision_parameters)

    with caplog.at_level(logging.ERROR, logger="hypermri.study"):
        study = Study(tmp_path, verbose=False, sort=None)

    assert list(study) == []
    assert any(
        "Failed to initialise experiment" in record.getMessage()
        and "Reading the parameter files" in record.getMessage()
        for record in caplog.records
    )
