"""Read the sequence mapping file and resolve raw method names to sequence IDs.

The packaged ``sequences/config/sequence_mapping.yaml`` names every sequence
by a lower-case canonical ID (such as ``"flash"``) and lists the raw scanner
method names it handles as method patterns:

* ``exact:<name>`` is an alias: one raw method name, compared ignoring case;
* ``contains:<text>`` matches every method name that contains ``text``;
* ``startswith:<text>`` matches every method name that starts with ``text``.

This module owns reading and validating that file and the matching rules.
The sequence registry (:mod:`hypermri.sequences.registry`) adds the Python
sequence classes; reconstruction asks :func:`method_name_to_sequence_id` for
the canonical ID only.  The module imports neither ``hypermri.sequences`` nor
``hypermri.reconstruction``, so both can depend on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from hypermri.config.mappings import check_fields, read_mapping_file
from hypermri.exceptions import ConfigurationError
from hypermri.loading.contracts import SequenceCategory

PACKAGED_SEQUENCE_MAPPING = (
    Path(__file__).parent / "sequences" / "config" / "sequence_mapping.yaml"
)
"""Path of the packaged sequence mapping file."""

METHOD_PATTERN_PREFIXES = ("exact:", "contains:", "startswith:")
"""Prefixes of a method pattern in the sequence mapping file.

``exact:`` names one raw method (an alias), ``contains:`` and ``startswith:``
match a family of methods.  Exact aliases take precedence over the other two;
see :func:`match_method_name`.
"""

_Entry = TypeVar("_Entry")


@dataclass(frozen=True)
class SequenceMappingEntry:
    """One validated entry of the sequence mapping file.

    Parameters
    ----------
    sequence_id : str
        Lower-case canonical sequence ID, the key of the entry in the file.
    category : SequenceCategory
        Category of the sequence.
    method_patterns : tuple of str
        Method patterns as written in the file, each starting with one of
        :data:`METHOD_PATTERN_PREFIXES` followed by a non-empty name.
    vendor : str or None
        Lower-case vendor name that restricts the entry, or ``None`` for an
        entry valid for all vendors.
    """

    sequence_id: str
    category: SequenceCategory
    method_patterns: tuple[str, ...]
    vendor: str | None = None


def sequence_mapping_error(
    problem: str, *, where: str, path: Path
) -> ConfigurationError:
    """Build the error for an invalid entry of a sequence mapping file.

    Parameters
    ----------
    problem : str
        What is wrong with the entry and how to correct it, as the end of a
        sentence without the final full stop.
    where : str
        Description of the entry, such as ``"sequence 'flash'"``.
    path : pathlib.Path
        The mapping file.

    Returns
    -------
    ConfigurationError
        The error, not raised; the caller raises it.
    """
    return ConfigurationError(
        f"{where.capitalize()} of the sequence mapping file {path} {problem}.",
        config_type="sequence",
        config_path=str(path),
    )


def load_sequence_mapping(path: Path) -> dict[str, SequenceMappingEntry]:
    """Read and validate a sequence mapping file.

    The file holds one section ``sequences`` that maps each lower-case
    canonical sequence ID to ``category`` (a
    :class:`~hypermri.loading.contracts.SequenceCategory` value),
    ``method_patterns`` (non-empty list of strings starting with ``exact:``,
    ``contains:`` or ``startswith:``) and optionally ``vendor`` (lower-case
    vendor name that restricts the entry).  The file names no Python classes:
    a custom sequence type is added in Python with
    :func:`hypermri.sequences.registry.register_sequence`.  Within one vendor scope, a canonical ID or an ``exact:`` alias identifies
    one sequence only; ``contains:`` and ``startswith:`` patterns may overlap
    other entries on purpose.

    Parameters
    ----------
    path : pathlib.Path
        The YAML file.  No built-in copy replaces a missing or invalid file.

    Returns
    -------
    dict of str to SequenceMappingEntry
        The entries keyed by canonical sequence ID, in file order.  The file
        is read anew on every call; no class is imported.

    Raises
    ------
    ConfigurationError
        If the file is missing or not valid YAML, repeats a key, has an
        unknown or missing section or field, has a ``class`` field, names a
        sequence or vendor not in lower case, has an unknown category or a
        method pattern without a known prefix, or maps one method name to two
        sequences.  The message names the file and the sequence.
    """
    path = Path(path)
    config = read_mapping_file(
        path, config_type="sequence", top_level_keys=("sequences",)
    )
    categories = [category.value for category in SequenceCategory]
    # Exact aliases and canonical IDs per vendor scope (None: all vendors).
    owners: dict[tuple, str] = {}
    entries: dict[str, SequenceMappingEntry] = {}

    for sequence_id, sequence_config in config["sequences"].items():
        where = f"sequence {sequence_id!r}"
        if isinstance(sequence_config, Mapping) and "class" in sequence_config:
            raise sequence_mapping_error(
                "has a 'class' field, but the mapping file no longer names "
                "Python classes; remove the field for a maintained sequence ID "
                "or add a custom sequence type in Python with "
                "hypermri.sequences.registry.register_sequence",
                where=where,
                path=path,
            )
        check_fields(
            sequence_config,
            required=("category", "method_patterns"),
            optional=("vendor",),
            where=where,
            config_type="sequence",
            path=path,
        )
        if not isinstance(sequence_id, str) or sequence_id != sequence_id.lower():
            raise sequence_mapping_error(
                "must be named in lower case", where=where, path=path
            )
        category = sequence_config["category"]
        if category not in categories:
            raise sequence_mapping_error(
                f"needs a category from {categories}, not {category!r}",
                where=where,
                path=path,
            )
        vendor = sequence_config.get("vendor")
        if vendor is not None and (
            not isinstance(vendor, str) or vendor != vendor.lower()
        ):
            raise sequence_mapping_error(
                "needs a lower-case vendor name", where=where, path=path
            )
        patterns = sequence_config["method_patterns"]
        if (
            not isinstance(patterns, list)
            or not patterns
            or not all(
                isinstance(pattern, str)
                and pattern.lower().startswith(METHOD_PATTERN_PREFIXES)
                and pattern.split(":", 1)[1]
                for pattern in patterns
            )
        ):
            raise sequence_mapping_error(
                "needs a non-empty list of method patterns, each starting "
                f"with one of {list(METHOD_PATTERN_PREFIXES)} followed by a "
                f"name, not {patterns!r}",
                where=where,
                path=path,
            )

        names = [sequence_id] + [
            pattern.split(":", 1)[1].lower()
            for pattern in patterns
            if pattern.lower().startswith("exact:")
        ]
        for name in dict.fromkeys(names):
            owner = owners.setdefault((vendor, name), sequence_id)
            if owner != sequence_id:
                raise sequence_mapping_error(
                    f"maps the method name {name!r}, which already identifies "
                    f"sequence {owner!r}; keep it under one sequence only",
                    where=where,
                    path=path,
                )

        entries[sequence_id] = SequenceMappingEntry(
            sequence_id=sequence_id,
            category=SequenceCategory(category),
            method_patterns=tuple(patterns),
            vendor=vendor,
        )
    return entries


_packaged_entries: dict[str, SequenceMappingEntry] | None = None


def get_packaged_sequence_mapping() -> dict[str, SequenceMappingEntry]:
    """Return the validated entries of the packaged sequence mapping file.

    Returns
    -------
    dict of str to SequenceMappingEntry
        The entries of :data:`PACKAGED_SEQUENCE_MAPPING` in file order.  The
        file is read on the first call and the result is cached for the
        interpreter; callers must not modify the returned dictionary.

    Raises
    ------
    ConfigurationError
        If the packaged file is missing or invalid; see
        :func:`load_sequence_mapping`.
    """
    global _packaged_entries
    if _packaged_entries is None:
        _packaged_entries = load_sequence_mapping(PACKAGED_SEQUENCE_MAPPING)
    return _packaged_entries


def _method_name_candidates(method_name: str) -> list[str]:
    """Return the raw and the scanner-path-free lower-case method name.

    Surrounding whitespace and quotes are removed; the second candidate is
    the part after the last ``/`` or ``\\``, as in ``%SiemensSeq%\\se``.
    """
    normalized = str(method_name).strip().strip('"').strip("'").lower()
    candidates = [normalized]

    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if basename and basename not in candidates:
        candidates.append(basename)

    return candidates


def _matches_patterns(
    method_name: str, patterns: Sequence[str], *, exact_only: bool
) -> bool:
    """Return whether a method name matches one of the method patterns.

    Patterns are compared ignoring case.  A pattern without a known prefix,
    which only runtime registration can add, is treated as ``contains:``.
    """
    method_lower = method_name.lower()

    for pattern in patterns:
        pattern_lower = pattern.lower()

        if pattern_lower.startswith("exact:"):
            if method_lower == pattern_lower[len("exact:") :]:
                return True
            continue

        if exact_only:
            continue
        if pattern_lower.startswith("startswith:"):
            if method_lower.startswith(pattern_lower[len("startswith:") :]):
                return True
        elif pattern_lower.startswith("contains:"):
            if pattern_lower[len("contains:") :] in method_lower:
                return True
        elif pattern_lower in method_lower:
            return True

    return False


def match_method_name(
    method_name: str,
    entry_groups: Sequence[Sequence[tuple[Sequence[str], _Entry]]],
) -> _Entry | None:
    """Return the first entry whose method patterns match a raw method name.

    The method name is compared ignoring case, both as given and without a
    scanner path prefix (the part after the last ``/`` or ``\\``).  All groups
    are first searched for an ``exact:`` alias, then again, in the same order,
    for any matching pattern, so a specific alias such as ``RAREVTR`` is not
    captured by a broader ``contains:rare`` pattern of another entry.

    Parameters
    ----------
    method_name : str
        Raw method name, such as ``"<Bruker:FLASH>"``.
    entry_groups : sequence of sequences of (patterns, entry) pairs
        Candidate entries in order of precedence, for example the
        vendor-specific entries before the entries valid for all vendors.
        Each entry comes with its method patterns.

    Returns
    -------
    object or None
        The first matching entry as given, or ``None`` if no pattern matches.
    """
    candidates = _method_name_candidates(method_name)
    for exact_only in (True, False):
        for group in entry_groups:
            for patterns, entry in group:
                if any(
                    _matches_patterns(candidate, patterns, exact_only=exact_only)
                    for candidate in candidates
                ):
                    return entry
    return None


def method_name_to_sequence_id(method_name: str | None) -> str | None:
    """Convert a raw method name to its canonical sequence ID.

    The lookup uses the packaged sequence mapping in three steps:

    1. a canonical ID or an ``exact:`` alias of any entry equal to
       ``method_name``, ignoring case;
    2. otherwise the first entry valid for all vendors whose ``exact:``
       alias, and then whose ``contains:`` or ``startswith:`` pattern,
       matches; see :func:`match_method_name`;
    3. otherwise ``None``.

    Step 1 resolves canonical IDs that no pattern names, such as
    ``"spin_echo"``.  Sequences added only at runtime through
    :func:`hypermri.sequences.registry.register_sequence` are not considered.

    Parameters
    ----------
    method_name : str or None
        Raw method name, such as ``"<Bruker:MGE>"``, or a canonical ID.

    Returns
    -------
    str or None
        The lower-case canonical sequence ID, or ``None`` if no entry matches
        or if ``method_name`` is ``None``.

    Raises
    ------
    ConfigurationError
        If the packaged sequence mapping cannot be read; see
        :func:`load_sequence_mapping`.

    Examples
    --------
    >>> method_name_to_sequence_id("<Bruker:FLASH>")
    'flash'
    >>> method_name_to_sequence_id("<User:my_flash_variant>")
    'flash'
    >>> method_name_to_sequence_id("unregistered_method") is None
    True
    """
    if method_name is None:
        return None

    entries = get_packaged_sequence_mapping()
    literal_names: dict[str, str] = {}
    for entry in entries.values():
        literal_names[entry.sequence_id] = entry.sequence_id
        for pattern in entry.method_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower.startswith("exact:"):
                literal_names[pattern_lower[len("exact:") :]] = entry.sequence_id

    sequence_id = literal_names.get(method_name.lower())
    if sequence_id is not None:
        return sequence_id

    general = [
        (entry.method_patterns, entry)
        for entry in entries.values()
        if entry.vendor is None
    ]
    entry = match_method_name(method_name, [general])
    return None if entry is None else entry.sequence_id
