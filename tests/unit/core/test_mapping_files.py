"""Validation of the packaged vendor and sequence mappings.

Each registry reads one YAML file.  The tests write small mapping files with
synthetic entries to ``tmp_path`` and check that an invalid file stops with a
``ConfigurationError`` naming the file and the entry, instead of being replaced
by a built-in copy or losing the entry silently.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from hypermri.exceptions import ConfigurationError
from hypermri.reconstruction.selection import RECONSTRUCTIONS
from hypermri.sequences.FLASH import FLASH
from hypermri.sequences.registry import SequenceRegistry
from hypermri.vendors.bruker import BrukerHandler
from hypermri.vendors.registry import VendorRegistry

_VALID = {
    VendorRegistry: {
        "vendors": {
            "bruker": {
                "priority": 100,
                "detection_patterns": ["file:fid", "contains:method"],
            }
        }
    },
    SequenceRegistry: {
        "sequences": {
            "flash": {
                "category": "anatomical",
                "method_patterns": ["exact:<Bruker:FLASH>", "contains:flash"],
            }
        }
    },
}
_REGISTRIES = list(_VALID)
# The path to the first entry of each mapping and the removed class-path field.
_FIRST_ENTRY = {
    VendorRegistry: (("vendors", "bruker"), "handler_class"),
    SequenceRegistry: (("sequences", "flash"), "class"),
}
_REGISTER_FUNCTION = {
    VendorRegistry: "register_vendor",
    SequenceRegistry: "register_sequence",
}


def _write(tmp_path: Path, config) -> Path:
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _first_entry(registry_class, config) -> dict:
    keys, _field = _FIRST_ENTRY[registry_class]
    entry = config
    for key in keys:
        entry = entry[key]
    return entry


def _registry_with(tmp_path, registry_class, change):
    config = copy.deepcopy(_VALID[registry_class])
    change(config)
    return registry_class(_write(tmp_path, config))


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_packaged_mapping_loads_and_validates(registry_class):
    registry = registry_class()

    assert registry.config_path.is_file()
    assert registry.list_registered()


def test_valid_synthetic_mappings_register_their_classes(tmp_path):
    vendors = VendorRegistry(_write(tmp_path, _VALID[VendorRegistry]))
    sequences = SequenceRegistry(_write(tmp_path, _VALID[SequenceRegistry]))

    assert vendors.get("bruker")["handler_class"] is BrukerHandler
    assert sequences.get_sequence_class("<Bruker:FLASH>") is FLASH


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_missing_mapping_file_raises_instead_of_using_a_builtin_copy(
    tmp_path, registry_class
):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="does not exist") as info:
        registry_class(missing)

    assert info.value.config_path == str(missing)


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_repeated_key_raises_instead_of_keeping_the_last_value(
    tmp_path, registry_class
):
    text = yaml.safe_dump(_VALID[registry_class], sort_keys=False)
    path = tmp_path / "mapping.yaml"
    path.write_text(text + text, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate key"):
        registry_class(path)


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_unknown_section_raises(tmp_path, registry_class):
    with pytest.raises(ConfigurationError, match=r"unknown fields \['settings'\]"):
        _registry_with(tmp_path, registry_class, lambda c: c.update(settings={}))


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_unknown_entry_field_raises_and_names_the_entry(tmp_path, registry_class):
    def add_field(config):
        _first_entry(registry_class, config)["clas"] = "typo"

    with pytest.raises(ConfigurationError, match=r"unknown fields \['clas'\]") as info:
        _registry_with(tmp_path, registry_class, add_field)

    keys, _field = _FIRST_ENTRY[registry_class]
    assert repr(keys[-1]) in str(info.value) or keys[-1] in str(info.value)
    assert "mapping.yaml" in str(info.value)


@pytest.mark.parametrize("registry_class", _REGISTRIES, ids=lambda c: c.__name__)
def test_class_path_field_is_rejected_with_the_python_alternative(
    tmp_path, registry_class
):
    """Regression: mapping files no longer import classes from dotted paths."""
    keys, field = _FIRST_ENTRY[registry_class]

    def add_class_path(config):
        _first_entry(registry_class, config)[field] = "hypermri.sequences.FLASH.FLASH"

    with pytest.raises(ConfigurationError, match=f"has a '{field}' field") as info:
        _registry_with(tmp_path, registry_class, add_class_path)

    message = str(info.value)
    assert repr(keys[-1]) in message
    assert "mapping.yaml" in message
    assert _REGISTER_FUNCTION[registry_class] in message


@pytest.mark.parametrize(
    "patterns",
    [["flash"], ["exact:"], [], "exact:flash"],
    ids=["no prefix", "empty name", "empty list", "not a list"],
)
def test_sequence_method_patterns_need_a_known_prefix(tmp_path, patterns):
    def set_patterns(config):
        config["sequences"]["flash"]["method_patterns"] = patterns

    with pytest.raises(ConfigurationError, match="method patterns"):
        _registry_with(tmp_path, SequenceRegistry, set_patterns)


def test_sequence_category_must_be_known(tmp_path):
    def set_category(config):
        config["sequences"]["flash"]["category"] = "imaging"

    with pytest.raises(ConfigurationError, match="'imaging'"):
        _registry_with(tmp_path, SequenceRegistry, set_category)


@pytest.mark.parametrize(
    "patterns",
    [["exact:<Bruker:FLASH>"], ["exact:flash"]],
    ids=["shared exact alias", "alias equals another id"],
)
def test_one_method_name_identifies_one_sequence(tmp_path, patterns):
    def add_sequence(config):
        config["sequences"]["other"] = {
            "category": "anatomical",
            "method_patterns": patterns,
        }

    with pytest.raises(ConfigurationError, match="already identifies sequence 'flash'"):
        _registry_with(tmp_path, SequenceRegistry, add_sequence)


def test_sequence_ids_are_lower_case(tmp_path):
    def rename(config):
        config["sequences"]["FLASH"] = config["sequences"].pop("flash")

    with pytest.raises(ConfigurationError, match="lower case"):
        _registry_with(tmp_path, SequenceRegistry, rename)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("detection_patterns", ["fid"], "detection patterns"),
        ("priority", True, "integer priority"),
        ("priority", "100", "integer priority"),
    ],
)
def test_vendor_fields_are_validated(tmp_path, field, value, message):
    def set_field(config):
        config["vendors"]["bruker"][field] = value

    with pytest.raises(ConfigurationError, match=message):
        _registry_with(tmp_path, VendorRegistry, set_field)


def test_packaged_strategy_vendors_are_registered_vendors():
    vendors = set(VendorRegistry().list_vendors())
    strategy_vendors = {vendor for vendor, _sequence, _kind in RECONSTRUCTIONS}

    assert strategy_vendors <= vendors


@pytest.mark.parametrize("patterns", [["file:"], [1], [None], "file:fid"])
def test_runtime_vendor_patterns_are_validated(patterns):
    registry = VendorRegistry()
    with pytest.raises(ValueError, match="Vendor"):
        registry.register_vendor("synthetic", object, patterns)
    assert registry.get_vendor_info("synthetic") is None


def test_runtime_unprefixed_vendor_pattern_is_rejected(tmp_path):
    registry = VendorRegistry()
    with pytest.raises(ValueError, match="needs detection patterns with prefixes"):
        registry.register_vendor("synthetic", object, ["synthetic"])
    assert registry.get_vendor_info("synthetic") is None
    with pytest.raises(ValueError, match="has none of the prefixes"):
        registry._matches_pattern(tmp_path / "synthetic-scan", "synthetic")


def test_vendor_detection_exposes_validator_defect_and_skips_io_errors(tmp_path):
    from hypermri.exceptions import VendorError

    class BrokenHandler:
        @staticmethod
        def is_valid_scan_dir(path):
            raise ValueError("synthetic validation defect")

    class UnreadableHandler:
        @staticmethod
        def is_valid_scan_dir(path):
            raise PermissionError("synthetic unreadable folder")

    registry = VendorRegistry()
    registry.register_vendor("broken", BrokenHandler, [], priority=1000)
    with pytest.raises(VendorError, match="BrokenHandler") as caught:
        registry.detect_vendor(tmp_path)
    assert isinstance(caught.value.__cause__, ValueError)
    registry.register_vendor("broken", UnreadableHandler, [], priority=1000)
    assert registry.detect_vendor(tmp_path) is None
