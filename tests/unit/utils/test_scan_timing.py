"""Check which scans ``load_scan_timing`` skips and which errors it reports.

The scans are invented stand-ins with ParaVision-style ``method`` and
``reco`` dictionaries; no scanner data is read. A scan without the timing
parameters is skipped from a collection with a warning, and a scan whose
timing values cannot be parsed fails with its name and the parameter.
"""

import logging

import pytest

from hypermri.utils.io.loaders import load_scan_timing


class _Scan:
    def __init__(self, name, method=None, reco=None):
        self.name = name
        self.method = method
        self.reco = reco

    def __str__(self):
        return self.name


def _timed_scan(name, duration="<0h0m2s0ms>", end="<2026-01-01T12:00:00,000+0100>"):
    return _Scan(
        name,
        method={"PVM_ScanTimeStr": duration, "PVM_Nucleus1": "<13C>"},
        reco={"RECO_time": end},
    )


def test_collection_keeps_readable_scans_in_order():
    scans = [
        _timed_scan("first", end="<2026-01-01T12:00:10,000+0100>"),
        _timed_scan("second", duration="<0h1m0s500ms>"),
    ]

    table = load_scan_timing(scans=scans)

    assert list(table["name"]) == ["first", "second"]
    durations = table.measurement_end_time_obj - table.measurement_start_time_obj
    assert [d.total_seconds() for d in durations] == [2.0, 60.5]


@pytest.mark.parametrize(
    ("scan", "missing"),
    [
        (_Scan("no_method", reco={"RECO_time": "<x>"}), "method"),
        (_Scan("no_duration", method={}, reco={"RECO_time": "<x>"}), "PVM_ScanTimeStr"),
        (
            _Scan("no_reco", method={"PVM_ScanTimeStr": "<0h0m2s0ms>"}),
            "reco",
        ),
        (
            _Scan("no_end", method={"PVM_ScanTimeStr": "<0h0m2s0ms>"}, reco={}),
            "RECO_time",
        ),
    ],
)
def test_collection_skips_scan_without_timing_with_a_warning(scan, missing, caplog):
    scans = [_timed_scan("kept"), scan]

    with caplog.at_level(logging.WARNING, logger="hypermri.utils.io.loaders"):
        table = load_scan_timing(scans=scans)

    assert list(table["name"]) == ["kept"]
    assert str(scan) in caplog.text
    assert missing in caplog.text


@pytest.mark.parametrize(
    ("scan", "missing"),
    [
        (_Scan("no_method", reco={"RECO_time": "<x>"}), "method"),
        (
            _Scan("no_end", method={"PVM_ScanTimeStr": "<0h0m2s0ms>"}, reco={}),
            "RECO_time",
        ),
    ],
)
def test_single_scan_without_timing_names_scan_and_parameter(scan, missing):
    with pytest.raises(KeyError, match=f"{scan}.*{missing}"):
        load_scan_timing(scan)


@pytest.mark.parametrize(
    ("scan", "parameter"),
    [
        (_timed_scan("bad_duration", duration="<2 s>"), "PVM_ScanTimeStr"),
        (_timed_scan("bad_end", end="<yesterday>"), "RECO_time"),
    ],
)
def test_unparsable_timing_fails_in_a_collection(scan, parameter):
    with pytest.raises(ValueError, match=f"{scan}.*{parameter}"):
        load_scan_timing(scans=[_timed_scan("kept"), scan])


def test_unparsable_timing_fails_for_a_single_scan():
    with pytest.raises(ValueError, match="bad_end.*RECO_time"):
        load_scan_timing(_timed_scan("bad_end", end="<yesterday>"))
