"""Resolve scanner method names to sequence classes and categories.

Notes
-----
Maintained types belong to the explicit Python sequence mapping; scanner method
patterns, categories and vendor scopes are validated external YAML settings.
Custom sequence types are added in Python through :func:`register_sequence`;
the mapping file names no classes. Registry construction imports definitions without constructing sequences or reading data.
Reading and validating the mapping file and matching method names belong to
:mod:`hypermri.sequence_mapping`; this module adds the sequence classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from hypermri.exceptions import ConfigurationError
from hypermri.loading.contracts import SequenceCategory
from hypermri.loading.protocols import BaseRegistry, SequenceProtocol
from hypermri.sequence_mapping import (
    PACKAGED_SEQUENCE_MAPPING,
    get_packaged_sequence_mapping,
    load_sequence_mapping,
    match_method_name,
    sequence_mapping_error,
)


class SequenceRegistry(BaseRegistry):
    """Registry of sequence classes and the raw method names they handle.

    The registry is built from one YAML mapping file whose ``sequences``
    section maps each lower-case canonical sequence ID to ``category`` (a
    :class:`~hypermri.loading.contracts.SequenceCategory` value),
    ``method_patterns`` (non-empty list of strings starting with ``exact:``,
    ``contains:`` or ``startswith:``) and optionally ``vendor`` (lower-case
    vendor name that restricts the entry). Each ID resolves through one
    explicit Python type mapping; a custom type is added with
    :meth:`register_sequence` after construction.

    The file is read and validated by
    :func:`~hypermri.sequence_mapping.load_sequence_mapping` during
    construction; the packaged file is read once per interpreter and shared
    with :func:`~hypermri.sequence_mapping.method_name_to_sequence_id`.
    Class definitions are imported
    without constructing sequences, reading acquired data, reconstructing or
    initializing viewers. Each registry owns its mutable method-pattern lists.
    No built-in copy replaces missing or invalid external settings.

    Within one vendor scope, a canonical ID or an ``exact:`` alias identifies
    one sequence only.  ``contains:`` and ``startswith:`` patterns may overlap
    other entries on purpose, because exact aliases are matched first and the
    remaining patterns in file order.

    Parameters
    ----------
    config_path : pathlib.Path, optional
        External settings file. The default is the packaged
        ``sequences/config/sequence_mapping.yaml``. Custom files may only
        use maintained sequence IDs.

    Raises
    ------
    ConfigurationError
        If the file is missing or not valid YAML, repeats a key, has an
        unknown or missing section or field, has a ``class`` field, names a
        sequence or vendor not in lower case, has an unknown category or a
        method pattern without a known prefix, maps one method name to two
        sequences, or names an ID without a maintained Python type. The
        message names the file and sequence.
    """

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.config_path = (
            PACKAGED_SEQUENCE_MAPPING if config_path is None else Path(config_path)
        )
        self._load_configurations()

    def register_sequence(
        self,
        sequence_name: str,
        sequence_class: Type[SequenceProtocol],
        category: SequenceCategory,
        method_patterns: List[str],
        vendor: Optional[str] = None,
    ) -> None:
        """
        Register a sequence class.

        Parameters
        ----------
            sequence_name: Name of the sequence (e.g., 'flash', 'csi')
            sequence_class: Sequence class implementing SequenceProtocol
            category: Sequence category (anatomical, spectroscopy, etc.)
            method_patterns: List of method name patterns that match this sequence
            vendor: Specific vendor (None for all vendors)
        """
        key = sequence_name.lower()
        if vendor:
            key = f"{vendor.lower()}.{key}"

        value = {
            "sequence_name": sequence_name.lower(),
            "sequence_class": sequence_class,
            "category": category,
            "method_patterns": method_patterns,
            "vendor": vendor,
        }
        self.register(key, value, overwrite=True)

    def get_sequence_class(
        self, method_name: str, vendor: Optional[str] = None
    ) -> Optional[Type[SequenceProtocol]]:
        """
        Get sequence class based on method name.

        Parameters
        ----------
            method_name: Method name from the scan
            vendor: Vendor name (for vendor-specific matching)

        Returns
        -------
            Sequence class or None if not found
        """
        if method_name is None:
            return None

        entry = self.resolve_sequence(method_name, vendor)
        if entry:
            return entry["sequence_class"]

        return None

    def resolve_sequence(
        self, method_name: str, vendor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve a raw method name to the registered sequence entry.

        Exact aliases are checked across the registry before broader
        ``contains:`` or ``startswith:`` patterns so specific variants such as
        ``RAREVTR`` are not captured by a more general ``RARE`` entry; see
        :func:`~hypermri.sequence_mapping.match_method_name`.  Entries of
        ``vendor`` take precedence over entries valid for all vendors; entries
        of other vendors are ignored.

        Parameters
        ----------
        method_name : str or None
            Raw method name, such as ``"<Bruker:FLASH>"`` or
            ``"%SiemensSeq%\\se"``, compared ignoring case.
        vendor : str, optional
            Vendor whose specific entries are searched first.

        Returns
        -------
        dict or None
            The registered entry with the keys ``sequence_name``,
            ``sequence_class``, ``category``, ``method_patterns`` and
            ``vendor``, or ``None`` if no entry matches or ``method_name`` is
            ``None``.  A canonical ID that no pattern names, such as
            ``"spin_echo"``, does not match.
        """
        if method_name is None:
            return None

        vendor_entries = []
        general_entries = []
        vendor_prefix = f"{vendor.lower()}." if vendor else None

        for key in self.list_registered():
            entry = self.get(key)
            if vendor_prefix and key.startswith(vendor_prefix):
                vendor_entries.append((entry["method_patterns"], entry))
            elif "." not in key:
                general_entries.append((entry["method_patterns"], entry))

        return match_method_name(method_name, [vendor_entries, general_entries])

    def list_sequences(
        self, category: Optional[SequenceCategory] = None, vendor: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        List registered sequences, optionally filtered.

        Parameters
        ----------
            category: Filter by sequence category
            vendor: Filter by vendor

        Returns
        -------
            Dictionary of sequence information
        """
        sequences = {}
        for key in self.list_registered():
            entry = self.get(key)

            # Apply filters
            if category and entry["category"] != category:
                continue
            if vendor and entry.get("vendor") != vendor:
                continue

            sequences[key] = entry

        return sequences

    def get_sequence_info(
        self, sequence_name: str, vendor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about a registered sequence."""
        key = sequence_name.lower()
        if vendor:
            key = f"{vendor.lower()}.{key}"
        return self.get(key)

    def _load_configurations(self) -> None:
        """Register every sequence of the validated mapping file.

        The validated entries of the packaged file are read once per
        interpreter; any other file is read anew.

        Raises
        ------
        ConfigurationError
            If the file is missing or invalid; see :class:`SequenceRegistry`.
        """
        entries = (
            get_packaged_sequence_mapping()
            if self.config_path == PACKAGED_SEQUENCE_MAPPING
            else load_sequence_mapping(self.config_path)
        )
        for sequence_name, entry in entries.items():
            where = f"sequence {sequence_name!r}"
            try:
                from ._sequence_types import _SEQUENCE_TYPES
            except (ImportError, AttributeError) as exc:
                raise ConfigurationError(
                    f"{where.capitalize()} of the sequence mapping file "
                    f"{self.config_path}: the Python sequence mapping cannot "
                    f"be imported: {exc}. Install the required sequence dependencies.",
                    config_type="sequence",
                    config_path=str(self.config_path),
                ) from exc
            sequence_class = _SEQUENCE_TYPES.get(sequence_name)
            if sequence_class is None:
                raise sequence_mapping_error(
                    "has no Python sequence mapping; use a maintained sequence "
                    "identifier, or add a custom sequence type in Python with "
                    "register_sequence",
                    where=where,
                    path=self.config_path,
                )

            self.register_sequence(
                sequence_name=sequence_name,
                sequence_class=sequence_class,
                category=entry.category,
                method_patterns=list(entry.method_patterns),
                vendor=entry.vendor,
            )


# Global registry instance
_sequence_registry = None


def get_sequence_registry() -> SequenceRegistry:
    """Get the global sequence registry instance."""
    global _sequence_registry
    if _sequence_registry is None:
        _sequence_registry = SequenceRegistry()
    return _sequence_registry


def register_sequence(
    sequence_name: str,
    sequence_class: Type[SequenceProtocol],
    category: SequenceCategory,
    method_patterns: List[str],
    vendor: Optional[str] = None,
) -> None:
    """Convenience function to register a sequence class."""
    registry = get_sequence_registry()
    registry.register_sequence(
        sequence_name, sequence_class, category, method_patterns, vendor
    )


def get_sequence_class(
    method_name: str, vendor: Optional[str] = None
) -> Optional[Type[SequenceProtocol]]:
    """Convenience function to get a sequence class."""
    registry = get_sequence_registry()
    return registry.get_sequence_class(method_name, vendor)
