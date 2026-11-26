"""Tests for the extra parameter columns of a ``Study``.

The study holds one synthetic ParaVision FLASH series without raw data.  The
parameter values are invented: ``PVM_RepetitionTime = 12.5`` and
``EffectiveTE = [1.5, 3.0]`` in ``method``, ``NR = 4`` in ``acqp``.
"""

from __future__ import annotations

import warnings

import pytest

from hypermri.study import Study


@pytest.fixture
def study(synthetic_flash_scan, write_paravision_parameters) -> Study:
    """Return a lazily loaded ``Study`` of one synthetic Bruker scan."""
    write_paravision_parameters(
        synthetic_flash_scan / "method",
        {
            "Method": "<Bruker:FLASH>",
            "PVM_RepetitionTime": 12.5,
            "EffectiveTE": [1.5, 3.0],
        },
    )
    write_paravision_parameters(
        synthetic_flash_scan / "acqp", {"ACQ_scan_name": "<synthetic>", "NR": 4}
    )
    return Study(synthetic_flash_scan, load_data="lazy", verbose=False, sort=None)


def _value(study: Study, column: str):
    return study.dataframe.iloc[0][column]


@pytest.mark.parametrize(
    ("options", "column", "expected"),
    [
        ({"keyword": "PVM_RepetitionTime"}, "PVM_RepetitionTime", 12.5),
        ({"keyword": "Method"}, "Method", "<Bruker:FLASH>"),
        ({"keyword": ("EffectiveTE", 1), "column_name": "TE 2"}, "TE 2", 3.0),
        ({"keyword": ("EffectiveTE", 0)}, "Parameter", 1.5),
        ({"keyword": ("EffectiveTE", 5)}, "Parameter", "MISSING"),
        ({"keyword": "NR", "paramfile": "acqp"}, "NR", 4),
        ({"keyword": "NR", "paramfile": ["method", "acqp"]}, "NR", 4),
        ({"keyword": "NR"}, "NR", "MISSING"),
        ({"keyword": "repetition_time_ms"}, "repetition_time_ms", 12.5),
        ({"keyword": "measurement_time"}, "measurement_time", "MISSING"),
        ({"keyword": "display_name"}, "display_name", "synthetic"),
        ({"keyword": {"bruker": "PVM_RepetitionTime"}}, "Parameter", 12.5),
        (
            {"keyword": {"siemens": "TR", "default": "PVM_RepetitionTime"}},
            "Parameter",
            12.5,
        ),
        (
            {
                "keyword": lambda scan: scan.handler.method["Method"],
                "column_name": "Sequence",
            },
            "Sequence",
            "<Bruker:FLASH>",
        ),
        ({"keyword": lambda scan: 1 / 0}, "Parameter", "MISSING"),
    ],
    ids=[
        "method-key",
        "method-name",
        "array-element",
        "array-element-default-name",
        "array-index-out-of-range",
        "acqp-key",
        "file-list",
        "key-in-another-file",
        "canonical-parameter",
        "canonical-parameter-without-value",
        "scan-property",
        "vendor-key",
        "default-key",
        "function",
        "failing-function",
    ],
)
def test_add_column_reads_the_value_without_a_warning(study, options, column, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study.add_column(**options)

    assert _value(study, column) == expected


def test_keywords_and_add_columns_fill_one_column_each(synthetic_flash_scan):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study = Study(
            synthetic_flash_scan,
            load_data="lazy",
            keywords=["Method"],
            verbose=False,
            sort=None,
        )
        study.add_columns(
            ["Method", ("Method", 0)], column_names=["Sequence", "First letter"]
        )

    assert _value(study, "Method") == "<Bruker:FLASH>"
    assert _value(study, "Sequence") == "<Bruker:FLASH>"
    assert _value(study, "First letter") == "<"


@pytest.mark.parametrize(
    "keyword",
    [
        {"siemens": "TR", "bruker": "PVM_RepetitionTime"},
        {"bruker": "PVM_RepetitionTime", "siemens": "TR"},
    ],
    ids=["bruker-last", "bruker-first"],
)
def test_vendor_keyword_mapping_uses_the_entry_of_the_scan_vendor(study, keyword):
    """Regression: the vendor name was always ``"str"``, so the first entry won."""
    study.add_column(keyword, column_name="TR")

    assert _value(study, "TR") == 12.5


def test_vendor_paramfile_mapping_uses_the_entry_of_the_scan_vendor(study):
    """Regression: the mapping fell back to ``"method"`` for every vendor."""
    study.add_column("NR", paramfile={"siemens": "header", "bruker": "acqp"})

    assert _value(study, "NR") == 4


def test_add_column_reads_a_handler_attribute_that_the_scan_does_not_define(study):
    scan = _value(study, "ScanObject")
    scan.handler.operator_note = "synthetic note"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study.add_column("operator_note")

    assert _value(study, "operator_note") == "synthetic note"


def test_add_column_rejects_a_parameter_file_that_no_scan_has(study):
    with pytest.raises(ValueError, match="paramfile") as error:
        study.add_column("NR", paramfile="acpq")

    message = str(error.value)
    assert "'acpq'" in message
    assert "'acqp'" in message  # the files that do exist are listed
    assert "NR" not in study.dataframe


def test_unknown_parameter_file_is_no_error_for_a_canonical_parameter(study):
    study.add_column("measurement_time", paramfile="acpq")

    assert _value(study, "measurement_time") == "MISSING"
