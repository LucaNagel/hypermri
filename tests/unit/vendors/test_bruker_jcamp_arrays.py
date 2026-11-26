"""Synthetic JCAMP records exercise wrapping without scanner metadata."""

from datetime import datetime, timezone

import numpy as np
import pytest

from hypermri.reconstruction.strategies.bruker.seq2d import _visu_parameter_text
from hypermri.vendors.bruker import BrukerHandler


def _read(tmp_path, text):
    path = tmp_path / "visu_pars"
    path.write_text(text, encoding="utf-8")
    handler = BrukerHandler.__new__(BrukerHandler)
    handler.path = tmp_path
    result = handler._read_param_file("visu_pars")
    assert path.read_bytes() == text.encode("utf-8")
    return result


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_complete_wrapped_string_arrays_preserve_delimiters_and_next_record(
    tmp_path, newline
):
    text = (
        "##$VisuFGElemComment=( 4, 100 )\n"
        "<first label> <> \n"
        "<a (tuple), key=value; escaped \\> delimiter> \n"
        "<hard-wra\npped label>\n"
        "##$Next=17\n"
        "##END=\n"
    ).replace("\n", newline)
    result = _read(tmp_path, text)
    assert result["VisuFGElemComment"] == (
        "<first label> <> <a (tuple), key=value; escaped \\> delimiter> "
        "<hard-wrapped label>"
    )
    assert result["Next"] == 17


def test_complete_wrapped_struct_arrays_retain_dependency_windows(tmp_path):
    result = _read(
        tmp_path,
        "##$VisuGroupDepVals=( 4 )\n"
        "(<VisuAcqEchoTime>, 0) (<VisuAcqRepetitionTime>, 0) \n"
        "(<VisuFGElemComment>, 0) \n"
        "(<VisuAcqInversionTime>, 0)\n"
        "##$Next=23\n",
    )
    assert _visu_parameter_text(result["VisuGroupDepVals"]) == (
        "(<VisuAcqEchoTime>, 0) (<VisuAcqRepetitionTime>, 0) "
        "(<VisuFGElemComment>, 0) (<VisuAcqInversionTime>, 0)"
    )
    assert result["Next"] == 23


def test_record_markers_inside_wrapped_strings_are_content(tmp_path):
    result = _read(
        tmp_path,
        "##$Labels=( 1, 100 )\n"
        "<text \n##$inside=value \n$$ inside string>\n"
        "##$Next=31\n",
    )
    assert result["Labels"] == "<text ##$inside=value $$ inside string>"
    assert result["Next"] == 31


def test_numeric_shapes_scalars_enums_and_timestamp_are_preserved(tmp_path):
    result = _read(
        tmp_path,
        "##$Table=( 2, 2 )\n1 2\n$$ layout comment\n3 4\n"
        "##$Scalar=( 1 )\n5\n"
        "##$Enums=( 2 )\nYes\nNo\n"
        "##$String=( 40 )\n<single string>\n"
        "##$Struct=(1, <two>, 3.5)\n"
        "$$ File finished by PARX at 2026-10-10 12:00:00.000 +0000\n"
        "##END=\n",
    )
    np.testing.assert_array_equal(result["Table"], [[1, 2], [3, 4]])
    assert result["Table"].dtype.kind == "i"
    assert result["Scalar"] == 5
    assert result["Enums"] == ["Yes", "No"]
    assert result["String"] == "<single string>"
    assert result["Struct"] == [1, "<two>", 3.5]
    assert result["MetaTimeStamp"] == datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


def test_empty_string_does_not_consume_next_parameter(tmp_path):
    result = _read(tmp_path, "##$Label=( 100 )\n<>\n##$Next=42\n")
    assert result == {"Label": "<>", "Next": 42}


def test_unquoted_equals_is_rejected_with_parameter_context(tmp_path):
    with pytest.raises(ValueError, match="RECO_transposition.*visu_pars"):
        _read(tmp_path, "##$RECO_transposition=invalid=parameter\n")


def test_equals_inside_scalar_string_and_struct_field_is_preserved(tmp_path):
    result = _read(tmp_path, "##$Label=<key=value>\n##$Struct=(1, <key=value>)\n")
    assert result == {"Label": "<key=value>", "Struct": [1, "<key=value>"]}
