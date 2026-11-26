"""Local fit monitor server: lazy start, page, encoded state and cleanup."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.visualization.fit_monitor import FitMonitor
from hypermri.visualization.render_data import PlotDataModel


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def decode(array: dict) -> np.ndarray:
    raw = base64.b64decode(array["b64"])
    dtype = {"float": "<f4", "complex": "<c8", "bool": np.uint8}[array["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(array["shape"])


def test_constructor_is_lazy_and_close_is_idempotent():
    monitor = FitMonitor()
    assert monitor.url is None and not monitor.running
    monitor.close()
    monitor.close()


def test_server_serves_page_and_latest_trace():
    trace = {
        "time_s": np.array([0.0, 0.001, 0.002]),
        "model_fid": np.array([[1 + 2j, 3 - 1j], [0.5j, -1]]),
        "active": np.array([True, False]),
        "reference": float("nan"),
        "objective": [1.0, float("inf"), -float("inf")],
        "count": np.int64(3),
        "name": "a",
    }
    with FitMonitor() as monitor:
        url = monitor.start()
        assert url == monitor.url and url.startswith("http://127.0.0.1:")
        assert b"FID Fit Monitor" in fetch(url)
        assert json.loads(fetch(url + "version"))["trace_version"] == 0

        monitor.publish(trace)
        monitor.publish_message("tracing x_idx=1", kind="busy")
        version = json.loads(fetch(url + "version"))
        assert version == {
            "trace_version": 1,
            "message": "tracing x_idx=1",
            "kind": "busy",
        }
        state = json.loads(fetch(url + "state"))
        sent = state["trace"]
        assert state["trace_version"] == 1
        np.testing.assert_allclose(decode(sent["time_s"]), trace["time_s"], rtol=1e-7)
        np.testing.assert_allclose(
            decode(sent["model_fid"]), trace["model_fid"], rtol=1e-7
        )
        np.testing.assert_array_equal(
            decode(sent["active"]).astype(bool), trace["active"]
        )
        assert sent["reference"] == {"__float__": "NaN"}
        assert sent["objective"][1:] == [
            {"__float__": "Infinity"},
            {"__float__": "-Infinity"},
        ]
        assert sent["count"] == 3 and sent["name"] == "a"
        with pytest.raises(urllib.error.HTTPError):
            fetch(url + "missing")
        with pytest.raises(ValueError, match="kind"):
            monitor.publish_message("x", kind="warning")
    assert monitor.url is None
    with pytest.raises(urllib.error.URLError):
        fetch(url + "version")
