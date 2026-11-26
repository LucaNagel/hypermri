"""Invented byte-level references for declared Bruker raw storage formats."""

import numpy as np
import pytest

from hypermri.vendors.bruker import BrukerHandler


def _handler(path, parameters):
    handler = BrukerHandler.__new__(BrukerHandler)
    handler.path = path
    handler.acqp = parameters
    handler.expnum = 1
    return handler


@pytest.mark.parametrize("filename", ["fid", "ser", "rawdata.job0"])
@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize(
    "format_name,code",
    [
        ("GO_16BIT_SGN_INT", "i2"),
        ("GO_32BIT_SGN_INT", "i4"),
        ("GO_32BIT_FLOAT", "f4"),
    ],
)
def test_raw_storage_decodes_declared_scalar_pairs(
    tmp_path, filename, endian, format_name, code
):
    """Small exactly representable scalar pairs decode independently of endian."""
    dtype = ("<" if endian == "little" else ">") + code
    payload = np.array([1, 2, 3, 4], dtype=dtype).tobytes()
    (tmp_path / filename).write_bytes(payload)
    handler = _handler(
        tmp_path,
        {"GO_raw_data_format": format_name, "BYTORDA": endian, "AQ_mod": "qdig"},
    )
    reader = (
        handler.load_rawdatajob0_file
        if filename == "rawdata.job0"
        else handler.load_fid_file
    )
    actual = reader()
    np.testing.assert_array_equal(actual, [1 + 2j, 3 + 4j])
    assert actual.dtype == np.complex128
    np.testing.assert_array_equal(reader(), actual)
    assert (tmp_path / filename).read_bytes() == payload


def test_rawdata_job_format_overrides_fid_format(tmp_path):
    """The job's explicit float64 storage declaration takes precedence."""
    (tmp_path / "rawdata.job0").write_bytes(
        np.array([1.25, 2.5, 3.75, 4.5], dtype=">f8").tobytes()
    )
    handler = _handler(
        tmp_path,
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "big",
            "AQ_mod": "qdig",
            "ACQ_ScanPipeJobSettings": "(No, STORE_64bit_float, No)",
        },
    )
    np.testing.assert_array_equal(
        handler.load_rawdatajob0_file(), [1.25 + 2.5j, 3.75 + 4.5j]
    )


def test_real_only_acquisition_is_not_paired(tmp_path):
    """AQ_mod=qf represents individual real samples, including odd counts."""
    (tmp_path / "fid").write_bytes(np.array([1, 2, 3], dtype="<i2").tobytes())
    handler = _handler(
        tmp_path,
        {"GO_raw_data_format": "GO_16BIT_SGN_INT", "BYTORDA": "little", "AQ_mod": "qf"},
    )
    np.testing.assert_array_equal(handler.load_fid_file(), [1, 2, 3])


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"BYTORDA": "unknown"}, "BYTORDA"),
        ({"GO_raw_data_format": "unknown"}, "GO_raw_data_format"),
        ({"AQ_mod": "qseq"}, "AQ_mod"),
    ],
)
def test_raw_storage_rejects_unsupported_declarations(tmp_path, overrides, match):
    """Unknown declarations cannot silently reinterpret the acquired bytes."""
    (tmp_path / "fid").write_bytes(np.array([1, 2], dtype="<i4").tobytes())
    handler = _handler(
        tmp_path,
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            **overrides,
        },
    )
    with pytest.raises(ValueError, match=match):
        handler.load_fid_file()


def test_complex_storage_rejects_incomplete_pair(tmp_path):
    """An incomplete complex pair is an error rather than a broadcast fallback."""
    (tmp_path / "fid").write_bytes(np.array([1, 2, 3], dtype="<i4").tobytes())
    handler = _handler(
        tmp_path,
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
        },
    )
    with pytest.raises(ValueError, match="pair"):
        handler.load_fid_file()


@pytest.mark.parametrize(
    "settings",
    [
        "(1, STORE_64bit_float, No)",
        ["(1, STORE_64bit_float, No)", "(1, STORE_32bit_signed, No)"],
    ],
)
def test_job_settings_are_decoded_from_actual_parameter_file(
    tmp_path, write_paravision_parameters, settings
):
    """Scalar structs and struct-array tokens retain the first job's dtype."""
    parameters = {
        "GO_raw_data_format": "GO_16BIT_SGN_INT",
        "BYTORDA": "little",
        "AQ_mod": "qdig",
        "ACQ_ScanPipeJobSettings": settings,
    }
    write_paravision_parameters(tmp_path / "acqp", parameters)
    handler = _handler(tmp_path, {})
    handler.acqp = handler._read_param_file("acqp")
    (tmp_path / "rawdata.job0").write_bytes(
        np.array([1.5, 2.25], dtype="<f8").tobytes()
    )
    np.testing.assert_array_equal(handler.load_rawdatajob0_file(), [1.5 + 2.25j])


@pytest.mark.parametrize("job0_index", [0, 1])
def test_named_job_selects_matching_storage_settings(
    tmp_path, write_paravision_parameters, job0_index
):
    """A named job0 need not be the first job in a PV360 parameter array."""
    jobs = ["(4, 1, 1, 1, 1, 1, 1, 1, <job1>)"] * 2
    jobs[job0_index] = "(4, 1, 1, 1, 1, 1, 1, 1, <job0>)"
    settings = ["(1, STORE_32bit_signed, No)"] * 2
    settings[job0_index] = "(1, STORE_64bit_float, No)"
    write_paravision_parameters(
        tmp_path / "acqp",
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            "ACQ_jobs": jobs,
            "ACQ_ScanPipeJobSettings": settings,
        },
    )
    handler = _handler(tmp_path, {})
    handler.acqp = handler._read_param_file("acqp")
    payload = np.array([1.5, 2.25], dtype="<f8").tobytes()
    (tmp_path / "rawdata.job0").write_bytes(payload)
    np.testing.assert_array_equal(handler.load_rawdatajob0_file(), [1.5 + 2.25j])
    assert (tmp_path / "rawdata.job0").read_bytes() == payload


@pytest.mark.parametrize(
    "jobs,settings,match",
    [
        (
            ["(4, 1, 1, 1, 1, 1, 1, 1, <job1>)"],
            ["(1, STORE_64bit_float, No)"],
            "exactly one",
        ),
        (
            ["(4, 1, 1, 1, 1, 1, 1, 1, <job0>)"] * 2,
            ["(1, STORE_64bit_float, No)"] * 2,
            "exactly one",
        ),
        (
            ["(4, 1, 1, 1, 1, 1, 1)"],
            ["(1, STORE_64bit_float, No)"],
            "Unsupported ACQ_jobs",
        ),
        (
            [
                "(4, 1, 1, 1, 1, 1, 1, 1, <job1>)",
                "(4, 1, 1, 1, 1, 1, 1, 1, <job0>)",
            ],
            ["(1, STORE_64bit_float, No)"],
            "Missing ACQ_ScanPipeJobSettings",
        ),
    ],
)
def test_job_storage_rejects_ambiguous_parameter_records(
    tmp_path, write_paravision_parameters, jobs, settings, match
):
    """Invalid job mappings fail before reading bytes with an unrelated dtype."""
    write_paravision_parameters(
        tmp_path / "acqp",
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            "ACQ_jobs": jobs,
            "ACQ_ScanPipeJobSettings": settings,
        },
    )
    handler = _handler(tmp_path, {})
    handler.acqp = handler._read_param_file("acqp")
    (tmp_path / "rawdata.job0").write_bytes(np.array([1, 2], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match=match):
        handler.load_rawdatajob0_file()


def test_older_jobs_use_positional_storage_settings(tmp_path):
    """Eight-field job records retain the older positional job0 contract."""
    handler = _handler(
        tmp_path,
        {
            "GO_raw_data_format": "GO_32BIT_SGN_INT",
            "BYTORDA": "little",
            "AQ_mod": "qdig",
            "ACQ_jobs": [[4, 1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1]],
            "ACQ_ScanPipeJobSettings": [
                [1, "STORE_64bit_float", "No"],
                [1, "STORE_32bit_signed", "No"],
            ],
        },
    )
    (tmp_path / "rawdata.job0").write_bytes(
        np.array([1.5, 2.25], dtype="<f8").tobytes()
    )
    np.testing.assert_array_equal(handler.load_rawdatajob0_file(), [1.5 + 2.25j])


@pytest.mark.parametrize("version", ["360.1.0", "<PV-360.1.0>"])
@pytest.mark.parametrize("word_size,code", [("_16_BIT", "i2"), ("_32_BIT", "i4")])
def test_pv360_word_size_without_legacy_format(tmp_path, version, word_size, code):
    """The supported PV360 declaration supplies storage precision directly."""
    handler = _handler(
        tmp_path,
        {
            "ACQ_sw_version": version,
            "ACQ_word_size": word_size,
            "BYTORDA": "big",
            "AQ_mod": "qdig",
        },
    )
    (tmp_path / "fid").write_bytes(np.array([1, 2], dtype=">" + code).tobytes())
    np.testing.assert_array_equal(handler.load_fid_file(), [1 + 2j])
