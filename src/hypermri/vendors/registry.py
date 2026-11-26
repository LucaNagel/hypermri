"""
Resolve Python vendor handler types using validated external detection settings.

Notes
-----
Canonical handler types belong to the Python vendor table; scanner filename
patterns and priorities are read from one YAML file per registry, which names
no Python classes; custom handler types are added through :func:`register_vendor`.
Registry construction reads settings and imports definitions without loading samples;
handler construction and acquired-data access follow the existing loading APIs.
"""

import threading
from pathlib import Path
from typing import Any

from hypermri.config.mappings import check_fields, read_mapping_file
from hypermri.exceptions import ConfigurationError, VendorError
from hypermri.loading.protocols import BaseRegistry, VendorHandlerProtocol

_DETECTION_PATTERN_PREFIXES = ("file:", "contains:", "extension:", "dir:")
"""Prefixes of a vendor detection pattern in ``vendor_mapping.yaml``."""

_PACKAGED_MAPPING = Path(__file__).parent / "config" / "vendor_mapping.yaml"
"""The vendor mapping file shipped with HyperMRI."""


def _validate_detection_patterns(patterns, vendor_name):
    """Validate one vendor's patterns for packaged and runtime registration."""
    if not isinstance(patterns, list):
        raise ValueError(
            f"Vendor {vendor_name!r} requires a list of detection patterns."
        )
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(
                f"Vendor {vendor_name!r} has an empty or non-string detection pattern."
            )
        if pattern.startswith(_DETECTION_PATTERN_PREFIXES):
            if not pattern.split(":", 1)[1]:
                raise ValueError(
                    f"Vendor {vendor_name!r} has an empty detection pattern {pattern!r}."
                )
        else:
            raise ValueError(
                f"Vendor {vendor_name!r} needs detection patterns with prefixes {_DETECTION_PATTERN_PREFIXES}, not {pattern!r}."
            )


class VendorRegistry(BaseRegistry):
    """Registry of vendor handlers and their detection patterns.

    The YAML file's vendors section maps lower-case vendor identifiers to
    priority (integer; higher values are tried first) and detection_patterns
    (strings starting with file:, contains:, extension: or dir:). Maintained
    identifiers resolve to one explicit Python handler-type table. The file
    names no Python classes; a custom handler type is added with
    register_vendor after construction.

    Parameters
    ----------
    config_path : pathlib.Path, optional
        The mapping file.  The default is the packaged
        ``vendors/config/vendor_mapping.yaml``.

    Raises
    ------
    ConfigurationError
        If the file is missing or not valid YAML, repeats a key, has an
        unknown or missing section or field, has a ``handler_class`` field,
        names a vendor not in lower case, has a non-integer priority or a
        detection pattern without a known prefix, or names a vendor without a
        maintained Python handler type. The message names the file and the
        vendor.

    Notes
    -----
    Construction eagerly reads/validates the file once and imports referenced
    definitions without constructing handlers or reading signal samples. No
    built-in settings replace a missing/invalid file. Registry lookup order,
    detection rules and handler options retain their existing behavior; entries
    are local mutable registration data. No numerical arrays, units, coordinates,
    storage precision or acquisition metadata are modified.

    Examples
    --------
    >>> registry = VendorRegistry()
    >>> registry.get("siemens")["handler_class"].__name__
    'SiemensHandler'
    """

    def __init__(self, config_path: Path | None = None):
        super().__init__()
        self.config_path = (
            _PACKAGED_MAPPING if config_path is None else Path(config_path)
        )
        self._load_configurations()

    def register_vendor(
        self,
        vendor_name: str,
        handler_class: type[VendorHandlerProtocol],
        detection_patterns: list[str],
        priority: int = 0,
    ) -> None:
        """Register validated runtime vendor detection settings.

        Parameters
        ----------
        vendor_name : str
            Nonempty vendor key, stored in lower case.
        handler_class : type
            Handler class used to load matching scans.
        detection_patterns : list of str
            Patterns with file:, contains:, extension: or dir: prefixes and a
            nonempty value.  An empty list registers a vendor that is detected
            only through its handler's ``is_valid_scan_dir``.
        priority : int
            Higher values are checked first. Boolean values are invalid.

        Returns
        -------
        None
            Updates this registry, replacing the existing vendor entry.

        Raises
        ------
        ValueError
            If the vendor name or priority is invalid, or a pattern is not a
            string, has no known prefix or has an empty value.
        """
        if not isinstance(vendor_name, str) or not vendor_name.strip():
            raise ValueError("register_vendor requires a nonempty vendor name.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(f"Vendor {vendor_name!r} requires an integer priority.")
        _validate_detection_patterns(detection_patterns, vendor_name)
        value = {
            "handler_class": handler_class,
            "detection_patterns": detection_patterns,
            "priority": priority,
        }
        self.register(vendor_name.lower(), value, overwrite=True)

    def get_vendor_handler(
        self,
        source: str | Path,
        load_data: bool = False,
        log_mode: str = "error",
        force_vendor: str | None = None,
        handler_options: dict[str, Any] | None = None,
    ) -> VendorHandlerProtocol | None:
        """Detect the vendor of a scan and construct its handler.

        Parameters
        ----------
        source : str or pathlib.Path
            Path to the scan.
        load_data : bool
            If true, the handler reads the data payload during construction;
            otherwise it reads only the parameters.
        log_mode : str
            Logging verbosity forwarded to the handler.
        force_vendor : str, optional
            Registered vendor name that bypasses detection.
        handler_options : dict, optional
            Extra keyword arguments for the handler constructor.

        Returns
        -------
        VendorHandlerProtocol or None
            The constructed handler, or ``None`` when no registered vendor
            recognizes ``source`` or ``force_vendor`` names no registered
            vendor.

        Raises
        ------
        Exception
            Whatever the handler constructor raises, for example
            ``FileNotFoundError`` or
            :class:`~hypermri.exceptions.DataLoadingError`; a scan that a
            vendor recognizes but cannot read is never reported as ``None``.
        """
        vendor_name = force_vendor or self.detect_vendor(source)
        if not vendor_name:
            return None

        final_options = self._prepare_handler_options(handler_options, log_mode)

        entry = self.get(vendor_name.lower())
        if entry:
            handler_class = entry["handler_class"]
            return handler_class(source=source, load_data=load_data, **final_options)
        return None

    def detect_vendor(self, path: str | Path) -> str | None:
        """
        Automatically detect vendor from path.

        Parameters
        ----------
        path : str or pathlib.Path
            File or directory to inspect without loading acquired samples.

        Returns
        -------
        str or None
            The first matching vendor in descending priority order, or None.

        Raises
        ------
        VendorError
            If a handler validator fails for a reason other than OSError.
            The handler name is included and the original error is the cause.

        Notes
        -----
        Unreadable folders are skipped. Detection reads directory metadata
        only and never constructs handlers or loads acquired data.

        """
        path = Path(path)

        # Get all vendors sorted by priority (descending)
        vendors = []
        for vendor_name in self.list_registered():
            entry = self.get(vendor_name)
            vendors.append((vendor_name, entry))

        vendors.sort(key=lambda x: x[1]["priority"], reverse=True)

        # First pass: ask each handler whether the path is a valid scan dir
        valid_matches = []
        for vendor_name, entry in vendors:
            handler_class = entry["handler_class"]

            if hasattr(handler_class, "is_valid_scan_dir"):
                try:
                    if handler_class.is_valid_scan_dir(path):
                        valid_matches.append((vendor_name, entry))
                except OSError:
                    continue
                except Exception as exc:
                    raise VendorError(
                        f"Vendor detection failed in {handler_class.__name__}.is_valid_scan_dir: {exc}",
                        vendor=vendor_name,
                    ) from exc

        if valid_matches:
            # Entries are already priority-sorted; prefer the first hit
            return valid_matches[0][0]

        # Second pass: fallback to pattern matching heuristics
        for vendor_name, entry in vendors:
            for pattern in entry["detection_patterns"]:
                try:
                    matches = self._matches_pattern(path, pattern)
                except OSError:
                    matches = False
                if matches:
                    return vendor_name

        return None

    def list_vendors(self) -> list[str]:
        """List all registered vendor names."""
        return self.list_registered()

    def get_vendor_info(self, vendor_name: str) -> dict[str, Any] | None:
        """Get information about a registered vendor."""
        return self.get(vendor_name.lower())

    def _matches_pattern(self, path: Path, pattern: str) -> bool:
        """Check if path matches detection pattern."""
        if pattern.startswith("extension:"):
            # File extension pattern
            ext = pattern.replace("extension:", "")
            return path.suffix.lower() == ext.lower()
        elif pattern.startswith("file:"):
            # Specific file pattern
            filename = pattern.replace("file:", "")
            return (
                (path / filename).exists() if path.is_dir() else path.name == filename
            )
        elif pattern.startswith("dir:"):
            # Directory pattern
            dirname = pattern.replace("dir:", "")
            return path.is_dir() and path.name == dirname
        elif pattern.startswith("contains:"):
            # Directory contains file pattern
            filename = pattern.replace("contains:", "")
            return path.is_dir() and any(p.name == filename for p in path.iterdir())
        raise ValueError(
            f"Detection pattern {pattern!r} has none of the prefixes "
            f"{_DETECTION_PATTERN_PREFIXES}; register it with one of them."
        )

    def _prepare_handler_options(
        self, handler_options: dict[str, Any] | None, log_mode: str
    ) -> dict[str, Any]:
        """Merge default registry kwargs with user overrides.

        Parameters
        ----------
        handler_options
            Optional mapping of user-provided handler keyword arguments.
        log_mode
            Logging verbosity hint propagated to the handler.

        Returns
        -------
        dict[str, Any]
            Combined handler keyword arguments.

        """
        options: dict[str, Any] = {"log_mode": log_mode}
        if handler_options:
            options.update(handler_options)
        return options

    def _load_configurations(self) -> None:
        """Register every vendor of the mapping file after validating it.

        Raises
        ------
        ConfigurationError
            If the file is missing or invalid; see :class:`VendorRegistry`.
        """
        config = read_mapping_file(
            self.config_path, config_type="vendor", top_level_keys=("vendors",)
        )
        for vendor_name, vendor_config in config["vendors"].items():
            where = f"vendor {vendor_name!r}"
            if isinstance(vendor_config, dict) and "handler_class" in vendor_config:
                self._raise_invalid(
                    where,
                    "has a 'handler_class' field, but the mapping file no longer "
                    "names Python classes; remove the field for a maintained "
                    "vendor or add a custom handler type in Python with "
                    "hypermri.vendors.registry.register_vendor",
                )
            check_fields(
                vendor_config,
                required=("priority", "detection_patterns"),
                optional=(),
                where=where,
                config_type="vendor",
                path=self.config_path,
            )
            if not isinstance(vendor_name, str) or vendor_name != vendor_name.lower():
                self._raise_invalid(where, "must be named in lower case")
            priority = vendor_config["priority"]
            if isinstance(priority, bool) or not isinstance(priority, int):
                self._raise_invalid(
                    where, f"needs an integer priority, not {priority!r}"
                )
            patterns = vendor_config["detection_patterns"]
            try:
                _validate_detection_patterns(patterns, vendor_name)
            except ValueError as exc:
                self._raise_invalid(where, str(exc))
            try:
                from ._handler_types import _VENDOR_HANDLERS
            except (ImportError, AttributeError) as exc:
                raise ConfigurationError(
                    f"{where.capitalize()} of the vendor mapping file "
                    f"{self.config_path}: the Python handler mapping cannot "
                    f"be imported: {exc}. Install the required vendor dependencies.",
                    config_type="vendor",
                    config_path=str(self.config_path),
                ) from exc

            handler_class = _VENDOR_HANDLERS.get(vendor_name)
            if handler_class is None:
                self._raise_invalid(
                    where,
                    "has no Python handler mapping; use a maintained vendor "
                    "identifier, or add a custom handler type in Python with "
                    "register_vendor",
                )
            self.register_vendor(
                vendor_name=vendor_name,
                handler_class=handler_class,
                detection_patterns=patterns,
                priority=priority,
            )

    def _raise_invalid(self, where: str, problem: str) -> None:
        """Raise the error for an invalid entry of the mapping file."""
        raise ConfigurationError(
            f"{where.capitalize()} of the vendor mapping file {self.config_path} "
            f"{problem}.",
            config_type="vendor",
            config_path=str(self.config_path),
        )


# Global registry instance with thread lock for safe lazy initialization
_vendor_registry = None
_vendor_registry_lock = threading.Lock()


def get_vendor_registry() -> VendorRegistry:
    """Get the global vendor registry instance."""
    global _vendor_registry
    if _vendor_registry is None:
        with _vendor_registry_lock:
            if _vendor_registry is None:
                _vendor_registry = VendorRegistry()
    return _vendor_registry


def register_vendor(
    vendor_name: str,
    handler_class: type[VendorHandlerProtocol],
    detection_patterns: list[str],
    priority: int = 0,
) -> None:
    """Register a vendor handler in the shared vendor registry.

    Delegates to :meth:`VendorRegistry.register_vendor` of the registry
    returned by :func:`get_vendor_registry`.

    Parameters
    ----------
    vendor_name : str
        Nonempty vendor key, stored in lower case.
    handler_class : type
        Handler class used to load matching scans.
    detection_patterns : list of str
        Patterns with file:, contains:, extension: or dir: prefixes and a
        nonempty value.
    priority : int, default 0
        Higher values are checked first.  Boolean values are invalid.

    Returns
    -------
    None
        The shared registry is updated in place and an existing entry for
        ``vendor_name`` is replaced; this affects every later detection in
        the process.

    Raises
    ------
    ValueError
        If the vendor name or priority is invalid, or a pattern is not a
        string, has no known prefix or has an empty value.

    Examples
    --------
    >>> from hypermri.vendors import register_vendor
    >>> register_vendor("myscanner", MyHandler, ["file:acqp.my"], priority=5)  # doctest: +SKIP
    """
    registry = get_vendor_registry()
    registry.register_vendor(vendor_name, handler_class, detection_patterns, priority)


def get_vendor_handler(
    source: str | Path,
    load_data: bool = False,
    log_mode: str = "error",
    force_vendor: str | None = None,
    handler_options: dict[str, Any] | None = None,
) -> VendorHandlerProtocol | None:
    """Detect the vendor of a scan and construct its handler.

    Delegates to :meth:`VendorRegistry.get_vendor_handler` of the shared
    registry.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to the scan.
    load_data : bool
        If true, the handler reads the data payload during construction;
        otherwise it reads only the parameters.
    log_mode : str
        Logging verbosity forwarded to the handler.
    force_vendor : str, optional
        Registered vendor name that bypasses detection.
    handler_options : dict, optional
        Extra keyword arguments for the handler constructor.

    Returns
    -------
    VendorHandlerProtocol or None
        The constructed handler, or ``None`` when no registered vendor
        recognizes ``source`` or ``force_vendor`` names no registered vendor.

    Raises
    ------
    Exception
        Whatever the handler constructor raises, for example
        ``FileNotFoundError`` or
        :class:`~hypermri.exceptions.DataLoadingError`.
    """
    registry = get_vendor_registry()
    return registry.get_vendor_handler(
        source=source,
        load_data=load_data,
        log_mode=log_mode,
        force_vendor=force_vendor,
        handler_options=handler_options,
    )


def detect_vendor(path: str | Path) -> str | None:
    """Convenience function to detect vendor from path."""
    registry = get_vendor_registry()
    return registry.detect_vendor(path)
