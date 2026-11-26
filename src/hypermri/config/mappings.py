"""Strict reading of the packaged YAML mapping files.

The vendor, sequence, reconstruction and plotter registries each own one YAML
file and its schema.  This module holds only what they share: reading a file
without silently dropping duplicate keys and checking the fields of one entry.
Every failure raises
:class:`~hypermri.exceptions.ConfigurationError` naming the file and the
entry, so a broken mapping stops loading instead of hiding an entry.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

from hypermri.exceptions import ConfigurationError


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects a key repeated within one mapping."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_mapping_file(
    path: Path, *, config_type: str, top_level_keys: Collection[str]
) -> dict[str, Any]:
    """Read a YAML mapping file and check its top-level sections.

    Parameters
    ----------
    path : pathlib.Path
        The YAML file.  It must exist; no built-in copy replaces a missing
        file.
    config_type : str
        Name of the mapping, such as ``"vendor"`` or ``"sequence"``, used in
        error messages.
    top_level_keys : collection of str
        The sections the file may contain.  Each one must be present and hold
        a mapping.

    Returns
    -------
    dict
        The parsed file.  It is read anew on every call.

    Raises
    ------
    ConfigurationError
        If the file is missing or not valid YAML, if a key is repeated within
        one mapping, if the top level is not a mapping, or if a section is
        missing, unknown or not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(
            f"The {config_type} mapping file {path} does not exist. Reinstall "
            "HyperMRI or pass the path of an existing mapping file.",
            config_type=config_type,
            config_path=str(path),
        )
    try:
        with open(path, encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"The {config_type} mapping file {path} is not valid YAML: {exc}. "
            "Each key may appear only once within a mapping.",
            config_type=config_type,
            config_path=str(path),
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            f"The {config_type} mapping file {path} must contain a mapping with "
            f"the sections {sorted(top_level_keys)}.",
            config_type=config_type,
            config_path=str(path),
        )
    check_fields(
        config,
        required=top_level_keys,
        where="the top level",
        config_type=config_type,
        path=path,
    )
    for section in top_level_keys:
        if not isinstance(config[section], dict):
            raise ConfigurationError(
                f"Section '{section}' of the {config_type} mapping file {path} "
                "must be a mapping.",
                config_type=config_type,
                config_path=str(path),
            )
    return config


def check_fields(
    entry: Any,
    *,
    required: Collection[str],
    optional: Collection[str] = (),
    where: str,
    config_type: str,
    path: Path,
) -> Mapping[str, Any]:
    """Check that one mapping entry has exactly the allowed fields.

    Parameters
    ----------
    entry : object
        The parsed entry.  It must be a mapping.
    required : collection of str
        Fields the entry must contain.
    optional : collection of str
        Fields the entry may contain in addition.
    where : str
        Description of the entry for error messages, such as
        ``"sequence 'flash'"``.
    config_type : str
        Name of the mapping, used in error messages.
    path : pathlib.Path
        The mapping file, used in error messages.

    Returns
    -------
    Mapping
        ``entry`` unchanged.

    Raises
    ------
    ConfigurationError
        If ``entry`` is not a mapping, lacks a required field or contains a
        field that is neither required nor optional.
    """
    if not isinstance(entry, Mapping):
        raise ConfigurationError(
            f"{where.capitalize()} of the {config_type} mapping file {path} must "
            f"be a mapping with the fields {sorted(required)}.",
            config_type=config_type,
            config_path=str(path),
        )
    missing = sorted(set(required) - set(entry))
    unknown = sorted(set(entry) - set(required) - set(optional))
    if missing or unknown:
        allowed = sorted(set(required) | set(optional))
        raise ConfigurationError(
            f"{where.capitalize()} of the {config_type} mapping file {path} has "
            f"missing fields {missing} and unknown fields {unknown}. Allowed "
            f"fields are {allowed}; required are {sorted(required)}.",
            config_type=config_type,
            config_path=str(path),
        )
    return entry
