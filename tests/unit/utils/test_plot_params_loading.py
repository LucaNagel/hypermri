"""Check which files ``load_plot_params`` reads and how it reports failures.

The parameter files are small invented JSON files in a temporary folder.
"""

import json
import logging
from types import SimpleNamespace

from hypermri.utils.io.files import load_plot_params


def test_parameters_are_read_from_the_given_folder(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"vmax": 3}))
    scan = SimpleNamespace()

    params = load_plot_params("params", path_to_param_file=tmp_path, scan=scan)

    assert params == {"vmax": 3}
    assert scan.plot_params == {"vmax": 3}


def test_parameter_file_path_is_the_fallback(tmp_path):
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps({"vmin": 1}))

    params = load_plot_params(str(param_file), path_to_param_file=tmp_path / "x")

    assert params == {"vmin": 1}


def test_missing_file_gives_empty_parameters_and_names_the_files(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hypermri.utils.io.files"):
        params = load_plot_params("absent", path_to_param_file=tmp_path)

    assert params == {}
    assert str(tmp_path / "absent.json") in caplog.text


def test_invalid_json_gives_empty_parameters_with_the_reason(tmp_path, caplog):
    (tmp_path / "params.json").write_text("{not json")

    with caplog.at_level(logging.WARNING, logger="hypermri.utils.io.files"):
        params = load_plot_params("params", path_to_param_file=tmp_path)

    assert params == {}
    assert "params.json" in caplog.text
