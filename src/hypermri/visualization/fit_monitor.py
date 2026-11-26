"""Serve a live browser page that follows one FID fit step by step.

:class:`FitMonitor` runs a small HTTP server on the local machine
(``127.0.0.1`` only) in a background thread. A notebook publishes the result
of :func:`~hypermri.analysis.spectroscopy.fit_spectrum_trace`; the page
(``fit_monitor_page.html``, packaged next to this module) polls the server and
draws the measured and model signals, the grid searches, frequency and T2*
inside their bounds, the complex amplitudes and the phase law of each step.
The page makes no scientific decision; it only renders the trace.
"""

from __future__ import annotations

import base64
import json
import math
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files as resource_files

import numpy as np

PAGE_RESOURCE = "fit_monitor_page.html"


class FitMonitor:
    """Local live page for fit traces, served from a background thread.

    Creating the object starts nothing; :meth:`start` binds the server.
    Published traces replace each other: the page always shows the latest.

    Parameters
    ----------
    host : str, optional
        Interface to bind. The default ``"127.0.0.1"`` keeps the page private
        to this computer.
    port : int, optional
        TCP port; ``0`` (default) lets the operating system choose a free one.

    Examples
    --------
    >>> from hypermri.analysis.spectroscopy import fit_spectrum_trace
    >>> monitor = FitMonitor()  # doctest: +SKIP
    >>> monitor.start()  # doctest: +SKIP
    'http://127.0.0.1:52344/'
    >>> monitor.publish(  # doctest: +SKIP
    ...     fit_spectrum_trace(data, fit_params=params, selection={"x_idx": 3})
    ... )
    >>> monitor.close()  # doctest: +SKIP
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._trace_version = 0
        self._state_json = _state_json(0, None)
        self._message = ("", "")

    @property
    def url(self) -> str | None:
        """str or None: Address of the page while running, else ``None``."""
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    @property
    def running(self) -> bool:
        """bool: Whether the server thread is serving requests."""
        return self._server is not None

    def start(self) -> str:
        """Start serving and return the page address.

        Returns
        -------
        str
            URL of the page, for example ``"http://127.0.0.1:52344/"``.
            Calling ``start`` again while running returns the same URL.

        Raises
        ------
        OSError
            If the port cannot be bound.
        """
        if self._server is not None:
            return self.url
        page = resource_files("hypermri.visualization").joinpath(PAGE_RESOURCE)
        page_bytes = page.read_bytes()
        server = ThreadingHTTPServer(
            (self._host, self._port), _handler(self, page_bytes)
        )
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="hypermri-fit-monitor", daemon=True
        )
        self._thread.start()
        return self.url

    def publish(self, trace: Mapping[str, object]) -> None:
        """Show a new trace on the page.

        Parameters
        ----------
        trace : mapping
            Output of :func:`~hypermri.analysis.spectroscopy.fit_spectrum_trace`.
            Arrays are sent as little-endian float32 (complex as interleaved
            real and imaginary float32), which is display precision only.
            The mapping is encoded immediately and not kept.
        """
        self._set(trace=_encode(trace), message=("", ""))

    def publish_message(self, text: str, *, kind: str = "") -> None:
        """Show a status line on the page and keep the current trace.

        Parameters
        ----------
        text : str
            Plain text such as ``"tracing x_idx=3 …"`` or an error message.
        kind : {"", "busy", "err"}, optional
            Styling of the line: neutral, work in progress or error.

        Raises
        ------
        ValueError
            If ``kind`` is not one of the listed values.
        """
        if kind not in ("", "busy", "err"):
            raise ValueError(
                f"FitMonitor.publish_message: kind={kind!r}; use '', 'busy' or 'err'."
            )
        self._set(message=(str(text), kind))

    def close(self) -> None:
        """Stop the server and release the port; safe to call repeatedly."""
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> FitMonitor:
        """Start the server and return the monitor."""
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        """Close the server."""
        self.close()

    def _set(self, *, message: tuple[str, str], trace=None) -> None:
        with self._lock:
            self._message = message
            if trace is not None:
                self._trace_version += 1
                self._state_json = _state_json(self._trace_version, trace)

    def _snapshot(self) -> tuple[int, str, tuple[str, str]]:
        with self._lock:
            return self._trace_version, self._state_json, self._message


def _state_json(trace_version: int, trace) -> str:
    """Serialize a published trace once, when it is published."""
    return json.dumps(
        {"trace_version": trace_version, "trace": trace},
        allow_nan=False,
        separators=(",", ":"),
    )


def _encode(value):
    """Turn a trace into JSON-ready values with base64 float32 arrays."""
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"__float__": "NaN"}
        return {"__float__": "Infinity" if value > 0 else "-Infinity"}
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _encode_array(array: np.ndarray) -> dict:
    """Encode an array of at most two dimensions for the page."""
    if array.ndim > 2:
        raise ValueError(
            f"FitMonitor: cannot send a {array.ndim}-D array; reduce it to 2-D."
        )
    if array.dtype == bool:
        dtype, raw = "bool", array.astype(np.uint8)
    elif np.iscomplexobj(array):
        dtype, raw = "complex", array.astype("<c8")
    elif array.dtype.kind in "fiu":
        dtype, raw = "float", array.astype("<f4")
    else:
        return [_encode(v) for v in array.tolist()]
    return {
        "__array__": True,
        "dtype": dtype,
        "shape": list(array.shape),
        "b64": base64.b64encode(np.ascontiguousarray(raw).tobytes()).decode("ascii"),
    }


def _handler(monitor: FitMonitor, page: bytes) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to one monitor."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - name required by http.server
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
                return
            trace_version, state, (message, kind) = monitor._snapshot()
            if path == "/version":
                body = json.dumps(
                    {"trace_version": trace_version, "message": message, "kind": kind}
                )
                self._send(body.encode(), "application/json")
            elif path == "/state":
                self._send(state.encode(), "application/json")
            else:
                self.send_error(404, "unknown path; open / for the fit monitor")

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002
            """Keep the notebook output free of request logs."""

    return Handler
