"""Tests of the measurement parameter rows shown by the browser viewer."""

import inspect
import re

import numpy as np

from hypermri.vendors.bruker import BrukerHandler
from hypermri.visualization.scan_parameters import parameter_rows


class _Scan:
    PARAMETER_SCHEMA = {
        "core": ("scan_name",),
        "acquisition": ("echo_time_ms", "flip_angle_deg", "missing"),
    }

    def __init__(self, parameters):
        self._parameters = parameters

    def get_parameters(self):
        return self._parameters


def test_rows_follow_the_schema_then_other_keys_sorted():
    scan = _Scan(
        {
            "zeta": 1,
            "echo_time_ms": 1.234567891,
            "scan_name": "csi",
            "alpha": True,
            "flip_angle_deg": None,
        }
    )
    assert parameter_rows(scan) == [
        {"group": "core", "name": "scan_name", "value": "csi", "source": ""},
        {
            "group": "acquisition",
            "name": "echo_time_ms",
            "value": "1.23457",
            "source": "",
        },
        {"group": "other", "name": "alpha", "value": "True", "source": ""},
        {"group": "other", "name": "zeta", "value": "1", "source": ""},
    ]


def test_arrays_and_long_texts_are_shortened():
    rows = parameter_rows(
        _Scan(
            {
                "short": np.array([1.0, 2.5]),
                "long": np.array(
                    [[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                ),
                "complex": np.ones(9, complex),
                "text": "x" * 500,
            }
        )
    )
    values = {row["name"]: row["value"] for row in rows}
    assert values["short"] == "[1, 2.5]"
    assert values["long"] == "array 3×3, 2 … 9"
    assert values["complex"] == "array 9"
    assert len(values["text"]) == 160 and values["text"].endswith("…")


def test_objects_without_parameters_give_no_rows():
    assert parameter_rows(None) == []
    assert parameter_rows(object()) == []
    assert parameter_rows(_Scan(None)) == []


def test_rows_name_the_vendor_parameter_of_the_scan_handler():
    class Handler:
        PARAMETER_SOURCES = {"echo_time_ms": "PVM_EchoTime"}

    scan = _Scan({"echo_time_ms": 1.0, "zeta": 2})
    scan.handler = Handler()
    sources = {row["name"]: row["source"] for row in parameter_rows(scan)}
    assert sources == {"echo_time_ms": "PVM_EchoTime", "zeta": ""}


def test_bruker_parameter_sources_are_read_by_the_bruker_parser():
    # The table is display only; every ParaVision name in it must occur in
    # the parser that reads the parameters, and every key must be canonical.
    parser = "".join(
        inspect.getsource(method)
        for method in (
            BrukerHandler._parse_parameters,
            BrukerHandler._get_measurement_type,
            BrukerHandler._get_measurement_time,
        )
    )
    sources = BrukerHandler.PARAMETER_SOURCES
    assert set(sources) <= BrukerHandler.parameter_keys()
    assert sources["matrix"] == "PVM_Matrix"
    for name in sources.values():
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", name):
            assert f'"{word}"' in parser, word
