"""Declared scalar storage formats shared by Bruker decoding stages."""

import re

import numpy as np


def _parameter_struct_records(value) -> list[list[str]]:
    """Normalize parsed scalar structs and struct arrays without losing records."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if all(isinstance(record, (list, tuple, np.ndarray)) for record in value):
            return [
                [str(field).strip().strip("<>") for field in record] for record in value
            ]
        values = np.asarray(value, dtype=object).reshape(-1)
        text = " ".join(str(field) for field in values)
    else:
        values = []
        text = str(value)
    records = re.findall(r"\(([^()]*)\)", text)
    if records:
        return [
            [field.strip().strip("<>") for field in record.split(",")]
            for record in records
        ]
    return [[str(field).strip().strip("<>") for field in values]] if len(values) else []


def _raw_storage_dtype(acqp: dict, *, rawdata_job: bool = False) -> np.dtype:
    """Resolve the declared scalar storage format without reading samples."""
    byte_order = acqp.get("BYTORDA")
    if byte_order not in {"little", "big"}:
        raise ValueError(
            f"Invalid Bruker BYTORDA: {byte_order!r}; expected little or big."
        )
    code = None
    if rawdata_job:
        settings = acqp.get("ACQ_ScanPipeJobSettings")
        if settings is not None:
            index = 0
            jobs = _parameter_struct_records(acqp.get("ACQ_jobs", []))
            if jobs:
                if all(len(job) == 9 for job in jobs):
                    matches = [i for i, job in enumerate(jobs) if job[8] == "job0"]
                    if len(matches) != 1:
                        raise ValueError(
                            "ACQ_jobs must identify exactly one rawdata.job0 record."
                        )
                    index = matches[0]
                elif not all(len(job) == 8 for job in jobs):
                    raise ValueError(
                        "Unsupported ACQ_jobs layout for rawdata.job0; expected eight-field positional or nine-field named records."
                    )
            records = _parameter_struct_records(settings)
            if index >= len(records):
                raise ValueError(
                    f"Missing ACQ_ScanPipeJobSettings record {index} for rawdata.job0."
                )
            fields = records[index]
            job_format = fields[1] if len(fields) > 1 else None
            code = {"STORE_32bit_signed": "i4", "STORE_64bit_float": "f8"}.get(
                job_format
            )
            if code is None:
                raise ValueError(
                    f"Unsupported rawdata.job0 storageDataType: {job_format!r}."
                )
    if code is None:
        format_name = acqp.get("GO_raw_data_format")
        code = {
            "GO_32BIT_SGN_INT": "i4",
            "GO_16BIT_SGN_INT": "i2",
            "GO_32BIT_FLOAT": "f4",
        }.get(format_name)
        if format_name is None and str(acqp.get("ACQ_sw_version", "")).strip(
            "<>"
        ).replace("PV-", "").replace("PV ", "").strip().startswith("360"):
            code = {"_32_BIT": "i4", "_16_BIT": "i2"}.get(acqp.get("ACQ_word_size"))
        if code is None:
            raise ValueError(
                f"Unsupported Bruker GO_raw_data_format/ACQ_word_size: {format_name!r}."
            )
    return np.dtype(("<" if byte_order == "little" else ">") + code)
