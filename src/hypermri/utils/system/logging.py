"""Configure package logging and write opt-in diagnostic files.

The ``hypermri`` logger owns the fallback stream handler. Descendants inherit
its threshold unless the application explicitly configures their levels.
Existing application handlers, formats and propagation settings take precedence.
Importing this module alone does not install a handler or write a file.
"""

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_LOG_SESSION_ID = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
_DEFAULT_FORMAT = "[%(levelname)s] %(name)s: %(message)s "
_DEFAULT_FORMATTERS: dict[str, logging.Formatter] = {}


class _ModuleFormatter(logging.Formatter):
    """Keep module-specific formats on the one package stream handler."""

    def format(self, record: logging.LogRecord) -> str:
        formatter = _DEFAULT_FORMATTERS.get(record.name)
        if formatter is not None:
            return formatter.format(record)
        return super().format(record)


def _use_default_stream(record: logging.LogRecord) -> bool:
    """Yield to application handlers on this record's propagation path."""
    logger = logging.getLogger(record.name)
    while logger is not None:
        if any(
            not getattr(handler, "_hypermri_default_stream", False)
            for handler in logger.handlers
        ):
            return False
        if not logger.propagate:
            break
        logger = logger.parent
    return True


def _ensure_package_stream_handler() -> None:
    """Install a fallback only when the package has no effective handler."""
    package_logger = logging.getLogger("hypermri")
    if not package_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler._hypermri_default_stream = True
        handler.setFormatter(_ModuleFormatter(_DEFAULT_FORMAT))
        handler.addFilter(_use_default_stream)
        package_logger.addHandler(handler)


def session_log_path(stem: str = "debug") -> str:
    """Return a process-session-specific path under the platform temp directory."""
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in stem
    ).strip("_")
    safe_stem = safe_stem or "debug"
    return str(
        Path(tempfile.gettempdir())
        / "hypermri"
        / "logs"
        / f"{safe_stem}-{_LOG_SESSION_ID}.log"
    )


@dataclass
class LogModesGenerator:
    critical: int = 50  # logging.CRITICAL
    error: int = 40  # logging.ERROR
    warning: int = 30  # logging.WARNING
    info: int = 20  # logging.INFO
    debug: int = 10  # logging.DEBUG
    notset: int = 0  # logging.NOTSET

    def __getitem__(cls, key):
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            return getattr(cls, key.lower())
        raise ValueError(f"'{key}' is not a valid logging mode.")

    def get(self, key, default=None):
        try:
            return self[key]
        except (AttributeError, ValueError):
            return default


LOG_MODES = LogModesGenerator()


def init_default_logger(
    name: str | None, default_mode: str | int = "error", fstring: str = _DEFAULT_FORMAT
) -> logging.Logger:
    """Return a logger with application-preserving default stream handling.

    Parameters
    ----------
    name : str or None
        Logger name. Names equal to ``hypermri`` or starting with ``hypermri.``
        share the package's fallback stream handler; other names use a local
        fallback only if they have no effective handler. None or an empty name
        selects the root logger, as in ``logging.getLogger``.
    default_mode : str or int, optional
        Case-insensitive name from ``LOG_MODES`` or a numeric logging level.
        Initializes an unset package level (or an unset non-package logger).
        Subsequent initialization preserves configured levels; package children
        inherit the package level unless explicitly configured by the caller.
    fstring : str, optional
        Percent-style log-record format for this logger's fallback output.
        Application handler formats are unchanged.

    Returns
    -------
    logging.Logger
        The named logger. Repeated calls return the same object and do not add
        handlers. Fallback output uses standard error.

    Raises
    ------
    AttributeError
        If a string level is not a member of ``LOG_MODES``.
    ValueError
        If the level type or the format string is unsupported.

    Notes
    -----
    This changes process-wide logging configuration but does not emit records.
    User handlers, their levels and formats, and propagation flags are preserved.
    The package fallback yields to application handlers on the record's path,
    including handlers installed after initialization.

    Examples
    --------
    >>> logger = init_default_logger("hypermri.study")
    >>> logger.name
    'hypermri.study'
    >>> set_hypermri_log_level("warning")
    """
    level = LOG_MODES[default_mode]
    formatter = logging.Formatter(fstring)
    logger = logging.getLogger(name)
    if logger.name == "hypermri" or logger.name.startswith("hypermri."):
        _DEFAULT_FORMATTERS[logger.name] = formatter
        package_logger = logging.getLogger("hypermri")
        if (
            not getattr(package_logger, "_hypermri_initialized", False)
            and package_logger.level == logging.NOTSET
        ):
            package_logger.setLevel(level)
        package_logger._hypermri_initialized = True
        _ensure_package_stream_handler()
    else:
        if logger.level == logging.NOTSET:
            logger.setLevel(level)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def set_hypermri_log_level(log_mode: str | int) -> None:
    """Set the shared package logging threshold without adding child handlers.

    Existing and subsequently imported child loggers inherit this threshold.
    Explicit child levels and application-owned logging configuration are
    preserved.

    Parameters
    ----------
    log_mode : str or int
        Case-insensitive name from ``LOG_MODES`` (for example ``"error"``)
        or a numeric logging level. ``"notset"`` restores root-level inheritance.

    Returns
    -------
    None
        Configuration is applied to the process-wide ``hypermri`` logger.

    Raises
    ------
    AttributeError
        If a string level is not a member of ``LOG_MODES``.
    ValueError
        If the level type is unsupported.

    Notes
    -----
    Repeated calls are idempotent for a given level. A single fallback stream
    handler is installed when no effective application handler exists. Handler
    levels and formatters, child logger levels and propagation, the root logger
    and unrelated namespaces remain unchanged. No log records are emitted.
    """
    package_logger = logging.getLogger("hypermri")
    package_logger.setLevel(LOG_MODES[log_mode])
    package_logger._hypermri_initialized = True
    _ensure_package_stream_handler()


def underoverline(text):
    """Overline and underline a string with dashes and print."""
    line = "-" * len(text)
    print(f"{line}\n{text}\n{line}")


def debug_log(
    message,
    file_name=None,
    use_timestamp=True,
    write_to_file=False,
    print_error=False,
    numpy_full_print=False,
):
    """
    Simple debug logging function that writes timestamped messages to a temporary file.

    Parameters
    ----------
    message : object or callable
        The message to log. A callable is evaluated only when file output is enabled.
    file_name : str or pathlib.Path, optional
        Destination file. Defaults to a process-session-specific platform temp path.
    use_timestamp : bool, optional
        Whether to prepend a timestamp to the message (default is True)
    write_to_file : bool, optional
        Whether to write the message to the file (default is False)
    print_error : bool, optional
        Whether to print an error message if writing fails (default is False)
    numpy_full_print : bool, optional
        Whether to enable full NumPy array printing (default is False)
    """
    if not write_to_file:
        return

    try:
        import datetime

        import numpy as np

        if callable(message):
            message = message()
        if numpy_full_print:
            with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=True):
                rendered_message = str(message)
        else:
            rendered_message = str(message)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        path = Path(file_name or session_log_path()).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            if use_timestamp:
                stream.write(f"[{timestamp}] {rendered_message}\n")
            else:
                stream.write(f"{rendered_message}\n")
            stream.flush()
    except Exception:
        if print_error:
            _LOGGER.exception("Failed to write debug log")
