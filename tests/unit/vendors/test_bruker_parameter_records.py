"""Invented JCAMP records exercise wrapping without scanner metadata."""

from datetime import datetime
from io import StringIO

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler


def _read_parameters(tmp_path, content):
    path = tmp_path / "method"
    path.write_text(content, encoding="utf-8")
    handler = object.__new__(BrukerHandler)
    handler.path = tmp_path
    return handler._read_param_file("method")


def test_multiline_string_array_preserves_labels_and_following_parameter(tmp_path):
    """String width is not an element count; wrapped contents remain exact."""
    result = _read_parameters(
        tmp_path,
        "##$VisuFGElemComment=( 4, 65 )\n"
        "<signal  intensity> <std dev of signal \n"
        "intensity> <> <Fit χ² (a,b)=value>\n"
        "##$Sentinel=17\n",
    )
    assert result["VisuFGElemComment"] == (
        "<signal  intensity> <std dev of signal intensity> <> <Fit χ² (a,b)=value>"
    )
    assert result["Sentinel"] == 17


def test_multiline_struct_array_preserves_every_dependency(tmp_path):
    result = _read_parameters(
        tmp_path,
        "##$VisuGroupDepVals=( 4 )\n"
        "(<VisuCoreOrientation>, 0) (<VisuCorePosition>, 0)\n"
        "(<VisuFGElemComment>, \n0) (<VisuAcqDiffusionBMatrix>, 0)\n"
        "##$Sentinel=23\n",
    )
    assert " ".join(result["VisuGroupDepVals"]) == (
        "(<VisuCoreOrientation>, 0) (<VisuCorePosition>, 0) "
        "(<VisuFGElemComment>, 0) (<VisuAcqDiffusionBMatrix>, 0)"
    )
    assert result["Sentinel"] == 23


def test_wrapped_array_ignores_internal_comments_and_retains_timestamp(tmp_path):
    result = _read_parameters(
        tmp_path,
        "##$Times=( 2, 2 )\n1 2\n$$ explanatory comment\n3 4\n"
        "$$ File finished by PARX at 2020-01-01 12:00:00.000 +0000\n"
        "\n##$Sentinel=<contains=equals>\n",
    )
    np.testing.assert_array_equal(result["Times"], [[1, 2], [3, 4]])
    assert result["MetaTimeStamp"] == datetime.fromisoformat(
        "2020-01-01T12:00:00+00:00"
    )
    assert result["Sentinel"] == "<contains=equals>"


@pytest.mark.parametrize(
    "declaration,body,expected",
    [
        ("( 1 )\n", "7\n", 7),
        ("( 2, 2 )\n", "1 2\n3 4\n", [[1, 2], [3, 4]]),
        ("( 3 )\n", "1.25\n2.5 3.75\n", [1.25, 2.5, 3.75]),
        ("( 65 )\n", "<one string>\n", "<one string>"),
        ("( 0 )\n", "", np.empty(0, dtype=int)),
    ],
)
def test_array_shapes_and_following_record_are_preserved(declaration, body, expected):
    stream = StringIO(body + "##$Next=99\n")
    result = BrukerHandler._parse_array(stream, declaration)
    if isinstance(expected, str):
        assert result == expected
    else:
        np.testing.assert_array_equal(result, expected)
    assert stream.readline() == "##$Next=99\n"


def test_incomplete_numeric_array_does_not_consume_next_parameter():
    stream = StringIO("1 2\n##$Next=99\n")
    with pytest.raises(ValueError):
        BrukerHandler._parse_array(stream, "( 3 )\n")
    assert stream.readline() == "##$Next=99\n"
