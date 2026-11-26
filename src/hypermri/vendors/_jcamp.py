"""JCAMP parameter parsing with explicit ParaVision and TopSpin conventions.

Both dialects share scalar conversion and physical value-block reading. Their
array shape and string contracts remain separate; no acquired file is modified.

Notes
-----
The ParaVision scalar, array and parameter readers adapt BrukerMRI's
ParseSingleValue, ParseArray and ReadParamFile operations, copyright 2014 Joerg
Doepfert, under Apache License 2.0. HyperMRI contributors modified path and
missing-file handling, tuple/timestamp and wrapped string/struct parsing,
record boundaries, and shared ParaVision/TopSpin primitives. See NOTICE and
LICENSES/brukerMRI-Apache-2.0.txt in the source distribution or the installed
distribution's dist-info/licenses directory for the attribution and licence.
"""

import io
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def _open_jcamp_text(file_path: Path) -> io.StringIO:
    """Decode a JCAMP parameter file independently of the platform locale.

    ParaVision and TopSpin write ASCII records; labels and comments may carry
    UTF-8 or Latin-1 characters. UTF-8 is tried first because it rejects
    invalid byte sequences; Latin-1 decodes every byte and serves as the
    fallback. Lines end in ``\\n`` after reading whatever the file used.

    Parameters
    ----------
    file_path : pathlib.Path
        Parameter file to read. The file is not modified.

    Returns
    -------
    io.StringIO
        Decoded text with universal newline translation.
    """
    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return io.StringIO(text, newline=None)


def _parse_scalar(val: str):
    """Extract a single value from a string.

    Parameters
    ----------
    value: str
        A string containing a single value.

    Returns
    -------
    An integer, a float, or a string without a newline character.

    Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

    """
    # check if int
    try:
        return int(val)
    except (TypeError, ValueError):
        pass

    # check if float
    try:
        return float(val)
    except (TypeError, ValueError):
        pass

    # otherwise remove newline character and return string
    return val.rstrip("\n")


def _read_value_lines(current_file, token_limit=None):
    value_lines = []
    in_string = False
    while token_limit is None or len(" ".join(value_lines).split()) < token_limit:
        position = current_file.tell()
        next_line = current_file.readline()
        if not next_line:
            break
        # A record boundary is not array data. Leave it for the file reader,
        # including the PARX timestamp trailer. Markers inside strings are data.
        if not in_string and (
            next_line.startswith("##")
            or next_line.startswith("$$ File finished by PARX at")
        ):
            current_file.seek(position)
            break
        if not in_string and next_line.lstrip().startswith("$$"):
            continue
        if (
            not in_string
            and value_lines
            and value_lines[-1].rstrip().endswith(")")
            and next_line.lstrip().startswith("(")
        ):
            value_lines.append(" ")
        value_lines.append(next_line)
        for match in re.finditer(r"\\.|[<>]", next_line):
            if match.group() == "<":
                in_string = True
            elif match.group() == ">":
                in_string = False

    return value_lines


def _parse_paravision_array(current_file, line):
    """
    Extract an array from a line in a file.

    Parameters
    ----------
    current_file: A file object to read from.
    line: str
        A string containing an array definition.

    Returns
    -------
    A shaped numeric NumPy array, a numeric scalar, a delimited string
    block or a list of enum/struct tokens. String dimensions include
    character capacity and are not interpreted as token counts. Physical
    wrapping is removed from strings and structs. The next record is left
    unread; acquired files are never modified.

    Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

    """
    array_size = tuple(int(size) for size in line.strip()[1:-1].split(","))
    value_lines = _read_value_lines(current_file)

    # ParaVision inserts physical line breaks without removing content.
    # Deleting the break rejoins wrapped strings and struct fields exactly.
    text = "".join(value.rstrip("\r\n") for value in value_lines).strip()
    if text.startswith("<"):
        # String dimensions include the maximum character count; they are
        # not a token count. Retain the handler's delimited-string contract.
        return text

    # Numeric/enum rows are whitespace separated even when the preceding
    # physical line has no trailing blank. Struct fields may be hard-wrapped.
    val_list = text.split() if text.startswith("(") else " ".join(value_lines).split()
    if not val_list:
        return np.array([]).reshape(array_size)
    try:
        float(val_list[0])
    except (TypeError, ValueError):
        return val_list[0] if len(val_list) == 1 else val_list

    try:
        values = [int(value) for value in val_list]
    except ValueError:
        values = [float(value) for value in val_list]
    return np.reshape(np.array(values), array_size) if len(values) > 1 else values[0]


def _read_paravision_parameters(file_path: Path) -> dict:
    """
    Reads a Bruker MRI experiment's method or acqp file to a dictionary.

    Parameters
    ----------
    file_path : pathlib.Path
        Parameter file path.

    Returns
    -------
    parsed_params: dict
        A dictionary containing the parsed parameters and their values.

    Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

    """
    param_dict = {}

    if not file_path.exists() or os.path.getsize(file_path) == 0:
        return param_dict

    with _open_jcamp_text(file_path) as f:
        while True:
            line = f.readline()
            if not line:
                break

            # when line contains parameter
            if line.startswith("##$"):
                (param_name, current_line) = line[3:].split("=", 1)
                unquoted = re.sub(r"<(?:\\.|[^>\\])*>", "", current_line)
                if "=" in unquoted:
                    raise ValueError(
                        f"Invalid JCAMP value for {param_name} in {file_path}: "
                        "an equals sign must be inside a <...> string"
                    )

                # if current entry (current_line) is arraysize
                if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                    value = _parse_paravision_array(f, current_line)

                # if current entry (current_line) is struct/list
                elif current_line[0] == "(" and current_line[-3:-1] != " )":
                    # if necessary read in multiple lines
                    while current_line[-2] != ")":
                        current_line = current_line[0:-1] + f.readline()

                    # parse the values to a list
                    value = [_parse_scalar(x) for x in current_line[1:-2].split(", ")]
                # otherwise current entry must be single string or number
                else:
                    value = _parse_scalar(current_line)

                # save parsed value to dict
                param_dict[param_name] = value

            # This exists only to extract the TimeStamp from the method
            # file
            if line.startswith("$$ File finished by PARX at"):
                time_str = line.split("at")[1].strip()
                format_str = "%Y-%m-%d %H:%M:%S.%f %z"

                time_obj = datetime.strptime(time_str, format_str)

                param_dict["MetaTimeStamp"] = time_obj

    return param_dict


def _parse_topspin_scalar(val):
    """Retain TopSpin's delimiter-stripping scalar convention."""
    return _parse_scalar(val.strip().strip("<>"))


def _parse_topspin_array(fh, size_line: str):
    """Parse a JCAMP-DX numeric or string array."""
    size_line = size_line.strip()[1:-1]  # remove ( )
    parts = size_line.split(",")
    array_size = 1
    for part in parts:
        part = part.strip()
        if ".." in part:
            lo, hi = part.split("..")
            array_size *= int(hi) - int(lo) + 1
        else:
            try:
                array_size *= int(part)
            except ValueError:
                pass

    values = " ".join(_read_value_lines(fh, token_limit=array_size)).split()

    if not values:
        return []

    if values[0].startswith("<"):
        return [v.strip("<>") for v in values]

    try:
        return np.array([int(v) for v in values])
    except ValueError:
        try:
            return np.array([float(v) for v in values])
        except ValueError:
            return values


def _read_topspin_parameters(file_path: Path) -> dict:
    """Parse a JCAMP-DX parameter file (``acqu`` / ``acqus``) into a dict."""
    if not file_path.exists() or os.path.getsize(file_path) == 0:
        return {}

    params: dict = {}
    with _open_jcamp_text(file_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break

            if not line.startswith("##$"):
                continue

            param_name, current_line = line[3:].split("=", 1)

            # Array with explicit size: ##$AMP= (0..31)\n100 100 ...
            if current_line.startswith("(") and current_line.rstrip().endswith(")"):
                value = _parse_topspin_array(fh, current_line)
            # Multi-line struct/list
            elif current_line.startswith("(") and ")" not in current_line:
                while ")" not in current_line:
                    next_line = fh.readline()
                    if not next_line:
                        break
                    current_line = current_line.rstrip("\n") + next_line
                value = [
                    _parse_topspin_scalar(x)
                    for x in current_line.strip()[1:-1].split(", ")
                ]
            else:
                value = _parse_topspin_scalar(current_line)

            params[param_name] = value

    return params
