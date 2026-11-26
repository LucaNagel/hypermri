"""Protect vendor classes and external detection settings; reject class paths."""

import builtins
import importlib
from importlib.resources import files

import pytest
import yaml

from hypermri.exceptions import ConfigurationError
from hypermri.vendors.registry import VendorRegistry

_EXPECTED = {
    "bruker": ("hypermri.vendors.bruker.BrukerHandler", 100),
    "bruker_topspin": ("hypermri.vendors.topspin.BrukerTopSpinHandler", 90),
    "siemens": ("hypermri.vendors.siemens.SiemensHandler", 50),
    "agilent": ("hypermri.vendors.agilent.AgilentHandler", 40),
    "magritek": ("hypermri.vendors.magritek.MagritekHandler", 45),
    "external": ("hypermri.vendors.external.ExternalDataHandler", -100),
}


def _class(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _write(tmp_path, vendor, **entry):
    path = tmp_path / "vendors.yaml"
    path.write_text(yaml.safe_dump({"vendors": {vendor: entry}}), encoding="utf-8")
    return path


@pytest.mark.parametrize("vendor", _EXPECTED)
def test_packaged_vendor_class_priority_and_order_are_preserved(vendor):
    registry = VendorRegistry()
    info = registry.get(vendor)
    path, priority = _EXPECTED[vendor]
    assert info["handler_class"] is _class(path)
    assert info["priority"] == priority
    assert registry.list_vendors() == list(_EXPECTED)


@pytest.mark.parametrize("vendor", _EXPECTED)
def test_external_settings_resolve_the_python_handler_type(tmp_path, vendor):
    path = _write(tmp_path, vendor, priority=7, detection_patterns=["file:invented"])
    registry = VendorRegistry(path)
    info = registry.get(vendor)
    assert info["handler_class"] is _class(_EXPECTED[vendor][0])
    assert info["priority"] == 7
    assert info["detection_patterns"] == ["file:invented"]


def test_packaged_vendor_resource_contains_only_external_settings():
    resource = files("hypermri.vendors").joinpath("config/vendor_mapping.yaml")
    config = yaml.safe_load(resource.read_text(encoding="utf-8"))
    assert set(config["vendors"]) == set(_EXPECTED)
    for entry in config["vendors"].values():
        assert set(entry) == {"priority", "detection_patterns"}


def test_unknown_vendor_without_a_python_type_fails_actionably(tmp_path):
    path = _write(tmp_path, "invented", priority=7, detection_patterns=[])
    with pytest.raises(
        ConfigurationError, match="invented.*Python handler mapping"
    ) as info:
        VendorRegistry(path)
    assert info.value.config_path == str(path)


@pytest.mark.parametrize("vendor", ["bruker", "invented"])
def test_handler_class_field_is_rejected_for_maintained_and_custom_vendors(
    tmp_path, vendor
):
    path = _write(
        tmp_path,
        vendor,
        handler_class="hypermri.vendors.external.ExternalDataHandler",
        priority=7,
        detection_patterns=[],
    )
    with pytest.raises(ConfigurationError, match="has a 'handler_class' field") as info:
        VendorRegistry(path)
    assert info.value.config_path == str(path)
    assert repr(vendor) in str(info.value)
    assert "register_vendor" in str(info.value)


def test_runtime_vendor_type_can_extend_classless_external_settings(tmp_path):
    path = _write(tmp_path, "external", priority=-100, detection_patterns=[])
    registry = VendorRegistry(path)
    handler = _class(_EXPECTED["external"][0])
    registry.register_vendor("invented", handler, ["file:invented"], priority=7)
    (tmp_path / "invented").touch()
    assert registry.detect_vendor(tmp_path) == "invented"


def test_repeated_external_vendor_key_is_rejected(tmp_path):
    path = tmp_path / "vendors.yaml"
    path.write_text("vendors:\n  external: {}\n  external: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate key 'external'"):
        VendorRegistry(path)


@pytest.mark.parametrize("field", ["priority", "detection_patterns"])
def test_external_settings_keep_required_fields(tmp_path, field):
    entry = {"priority": 7, "detection_patterns": []}
    del entry[field]
    path = _write(tmp_path, "external", **entry)
    with pytest.raises(ConfigurationError, match=f"missing fields.*{field}"):
        VendorRegistry(path)


def test_external_settings_reject_unknown_fields(tmp_path):
    path = _write(tmp_path, "external", priority=7, detection_patterns=[], clas="typo")
    with pytest.raises(ConfigurationError, match="unknown fields.*clas"):
        VendorRegistry(path)


def test_python_handler_import_failure_keeps_the_configuration_error_contract(
    tmp_path, monkeypatch
):
    path = _write(tmp_path, "external", priority=-100, detection_patterns=[])
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "_handler_types":
            raise ImportError("invented missing vendor dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(
        ConfigurationError, match="Python handler mapping cannot be imported"
    ) as info:
        VendorRegistry(path)
    assert isinstance(info.value.__cause__, ImportError)
    assert info.value.config_path == str(path)
