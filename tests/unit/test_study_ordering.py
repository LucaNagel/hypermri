"""Tests for the order of scans in a ``Study``.

The scans are synthetic ParaVision FLASH series without raw data.  Their
acquisition order is set only through the line ``$$ File finished by PARX at
...`` of the ``method`` file (parsed into ``handler.timestamp``) and through
``ACQ_time`` in ``acqp`` (parsed into ``handler.measurement_time``).  All
times are invented; no scanner data are involved.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hypermri.study import Study, _sorted_by_acquisition_time

_VISU_PARS = {
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


@pytest.fixture
def write_study(tmp_path, write_paravision_parameters):
    """Return a writer of a synthetic ParaVision study directory.

    The writer takes a mapping from experiment number to a dictionary with the
    optional entries ``"finished"`` (clock time ``HH:MM:SS`` written to the
    ``method`` file as the PARX time stamp of 1 March 2024, +01:00) and
    ``"acq_time"`` (clock time written as ``ACQ_time`` of the same day).  An
    absent entry leaves the corresponding time out of the files.
    """

    def write(scans: dict[int, dict[str, str]]) -> Path:
        study = tmp_path / "study"
        for number, times in scans.items():
            scan = study / str(number)
            write_paravision_parameters(scan / "method", {"Method": "<Bruker:FLASH>"})
            if "finished" in times:
                with open(scan / "method", "a", encoding="utf-8") as method:
                    method.write(
                        "$$ File finished by PARX at "
                        f"2024-03-01 {times['finished']}.000 +0100\n"
                    )
            acqp = {"ACQ_scan_name": f"<synthetic {number}>"}
            if "acq_time" in times:
                acqp["ACQ_time"] = f"<2024-03-01T{times['acq_time']},000+0100>"
            write_paravision_parameters(scan / "acqp", acqp)
            write_paravision_parameters(scan / "visu_pars", _VISU_PARS)
            write_paravision_parameters(scan / "pdata" / "1" / "visu_pars", _VISU_PARS)
            write_paravision_parameters(
                scan / "pdata" / "1" / "reco", {"RECO_transposition": [0, 0]}
            )
            (scan / "pdata" / "1" / "2dseq").write_bytes(
                np.arange(8, dtype="<i2").tobytes()
            )
        return study

    return write


def _order(path: Path, sort="time") -> list:
    return list(Study(path, verbose=False, sort=sort, load_data="lazy").dataframe.index)


def test_time_sort_orders_scans_by_method_time_stamp(write_study):
    path = write_study(
        {
            1: {"finished": "12:00:00"},
            2: {"finished": "10:00:00"},
            3: {"finished": "11:00:00"},
        }
    )

    assert _order(path) == [2, 3, 1]
    assert _order(path, sort=None) == [1, 2, 3]


def test_time_sort_keeps_experiment_order_for_equal_time_stamps(write_study):
    scans = {number: {"finished": "10:00:00"} for number in range(1, 21)}
    scans[7] = {"finished": "09:00:00"}

    assert _order(write_study(scans)) == [7, *range(1, 7), *range(8, 21)]


def test_time_sort_places_scans_without_any_time_last(write_study):
    path = write_study(
        {
            1: {},
            2: {"finished": "12:00:00"},
            3: {},
            4: {"finished": "10:00:00"},
        }
    )

    assert _order(path) == [4, 2, 1, 3]


def test_time_sort_uses_acquisition_time_without_method_time_stamp(write_study):
    path = write_study(
        {
            1: {"finished": "12:00:00"},
            2: {"acq_time": "11:00:00"},
            3: {"finished": "10:00:00"},
        }
    )

    assert _order(path) == [3, 2, 1]


@pytest.mark.parametrize(
    ("sort", "expected"),
    [
        ("time", [2, 3, 1]),
        ("e_number", [1, 2, 3]),
        ("sequence_class", [1, 2, 3]),
        ("scan_name", [1, 2, 3]),
        (None, [1, 2, 3]),
    ],
)
def test_sort_modes_and_removal_emit_no_warning(write_study, sort, expected):
    path = write_study(
        {
            1: {"finished": "12:00:00"},
            2: {"finished": "10:00:00"},
            3: {"finished": "11:00:00"},
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study = Study(path, verbose=False, sort=sort, load_data="lazy")
        assert list(study.dataframe.index) == expected
        study.remove(3)

    assert list(study.dataframe.index) == [key for key in expected if key != 3]


def test_time_sort_compares_time_zone_aware_and_naive_times():
    """A naive time is read as local time, as ``datetime.timestamp`` does."""
    local = datetime(2024, 3, 1, 12, 0, 0)
    scans = [
        SimpleNamespace(handler=SimpleNamespace(timestamp=local)),
        SimpleNamespace(
            handler=SimpleNamespace(
                timestamp=(local - timedelta(hours=1)).astimezone(timezone.utc)
            )
        ),
        SimpleNamespace(handler=None),
        SimpleNamespace(
            handler=SimpleNamespace(
                timestamp=None, measurement_time=local - timedelta(hours=2)
            )
        ),
    ]
    table = pd.DataFrame({"ScanObject": scans}, index=pd.Index([1, 2, 3, 4]))

    ordered = _sorted_by_acquisition_time(table)

    assert list(ordered.index) == [4, 2, 1, 3]
    assert list(table.index) == [1, 2, 3, 4]
