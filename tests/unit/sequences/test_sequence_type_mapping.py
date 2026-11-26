"""Preserve sequence identities and external aliases; reject class paths."""

import builtins
import importlib
from importlib.resources import files

import pytest
import yaml

from hypermri.exceptions import ConfigurationError
from hypermri.loading.contracts import SequenceCategory
from hypermri.reconstruction.csi_raw import (
    FOURIER_ENCODED_METHODS,
    SCANNER_SPATIAL_METHODS,
)
from hypermri.sequences.registry import SequenceRegistry

_EXPECTED = {
    "flash": ("hypermri.sequences.FLASH.FLASH", "anatomical"),
    "bssfp": ("hypermri.sequences.BSSFP.BSSFP", "anatomical"),
    "b1calibration": ("hypermri.sequences.B1Calibration.B1Calibration", "anatomical"),
    "dtiepi": ("hypermri.sequences.DTIEPI.DTIEPI", "anatomical"),
    "epi": ("hypermri.sequences.EPI.EPI", "anatomical"),
    "fair_rare": ("hypermri.sequences.FAIR_RARE.FAIR_RARE", "anatomical"),
    "fexi": ("hypermri.sequences.fexi.FEXI", "diffusion"),
    "fieldmap": ("hypermri.sequences.Fieldmap.Fieldmap", "anatomical"),
    "mge": ("hypermri.sequences.MGE.MGE", "anatomical"),
    "rarevtr": ("hypermri.sequences.RAREVTR.RAREVTR", "anatomical"),
    "rare": ("hypermri.sequences.RARE.RARE", "anatomical"),
    "spin_echo": ("hypermri.sequences.SpinEcho.SpinEcho", "anatomical"),
    "csi": ("hypermri.sequences.CSI.CSI", "spectroscopy"),
    "press": ("hypermri.sequences.PRESS.PRESS", "spectroscopy"),
    "steam": ("hypermri.sequences.STEAM.STEAM", "spectroscopy"),
    "singlepulse": ("hypermri.sequences.Singlepulse.Singlepulse", "spectroscopy"),
    "gnspect_ms": ("hypermri.sequences.gNSPECT_MS.gNSPECT_MS", "spectroscopy"),
    "nspect": ("hypermri.sequences.NSPECT.NSPECT", "spectroscopy"),
    "geoffslcspec": ("hypermri.sequences.GeoffSlcSpec.GeoffSlcSpec", "spectroscopy"),
    "ivrecovery": ("hypermri.sequences.IVRecovery.IVRecovery", "spectroscopy"),
    "semilaser": ("hypermri.sequences.semiLASER.semiLASER", "spectroscopy"),
    "cpmg": ("hypermri.sequences.CPMG.CPMG", "spectroscopy"),
    "msme": ("hypermri.sequences.MSME.MSME", "anatomical"),
    "mdeft": ("hypermri.sequences.MDEFT.MDEFT", "anatomical"),
    "flowmap": ("hypermri.sequences.FLOWMAP.FLOWMAP", "anatomical"),
    "dti": ("hypermri.sequences.DTI.DTI", "anatomical"),
    "spiral": ("hypermri.sequences.SPIRAL.SPIRAL", "anatomical"),
    "dtispiral": ("hypermri.sequences.DTISPIRAL.DTISPIRAL", "anatomical"),
    "ute": ("hypermri.sequences.UTE.UTE", "anatomical"),
    "rfprofile": ("hypermri.sequences.RFPROFILE.RFPROFILE", "anatomical"),
    "isis": ("hypermri.sequences.ISIS.ISIS", "spectroscopy"),
    "zte": ("hypermri.sequences.ZTE.ZTE", "anatomical"),
}


def _class(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _write(tmp_path, entries):
    path = tmp_path / "sequences.yaml"
    path.write_text(
        yaml.safe_dump({"sequences": entries}, sort_keys=False), encoding="utf-8"
    )
    return path


def _entry(**overrides):
    return {
        "category": "anatomical",
        "method_patterns": ["exact:invented"],
        **overrides,
    }


@pytest.mark.parametrize("sequence", _EXPECTED)
def test_packaged_sequence_identity_category_and_order_are_preserved(sequence):
    registry = SequenceRegistry()
    info = registry.get_sequence_info(sequence)
    path, category = _EXPECTED[sequence]
    assert info["sequence_class"] is _class(path)
    assert info["category"] is SequenceCategory(category)
    assert info["vendor"] is None
    assert registry.list_registered() == list(_EXPECTED)


@pytest.mark.parametrize("sequence", _EXPECTED)
def test_classless_external_settings_resolve_each_python_sequence_type(
    tmp_path, sequence
):
    path, category = _EXPECTED[sequence]
    registry = SequenceRegistry(_write(tmp_path, {sequence: _entry(category=category)}))
    info = registry.resolve_sequence("invented")
    assert info["sequence_class"] is _class(path)
    assert info["category"] is SequenceCategory(category)
    assert info["method_patterns"] == ["exact:invented"]


def test_packaged_resource_contains_only_external_sequence_settings():
    resource = files("hypermri.sequences").joinpath("config/sequence_mapping.yaml")
    config = yaml.safe_load(resource.read_text(encoding="utf-8"))
    assert list(config["sequences"]) == list(_EXPECTED)
    for entry in config["sequences"].values():
        assert set(entry) == {"category", "method_patterns"}


def test_named_scanner_csi_methods_are_the_ones_with_a_reconstruction():
    """Regression: ``<User:lucaCSIReco>`` was listed without a reconstruction.

    A ParaVision method (``<Bruker:...>`` or ``<User:...>``) that the mapping
    names for CSI must be one the CSI reconstruction supports, and the reverse.
    """
    resource = files("hypermri.sequences").joinpath("config/sequence_mapping.yaml")
    config = yaml.safe_load(resource.read_text(encoding="utf-8"))
    named = {
        pattern.split(":", 1)[1]
        for pattern in config["sequences"]["csi"]["method_patterns"]
        if pattern.startswith("exact:<")
    }

    assert named == FOURIER_ENCODED_METHODS | SCANNER_SPATIAL_METHODS


@pytest.mark.parametrize("sequence", ["flash", "invented"])
def test_class_path_field_is_rejected_for_maintained_and_custom_ids(tmp_path, sequence):
    path = _write(tmp_path, {sequence: _entry(**{"class": _EXPECTED["csi"][0]})})
    with pytest.raises(ConfigurationError, match="has a 'class' field") as info:
        SequenceRegistry(path)
    assert info.value.config_path == str(path)
    assert repr(sequence) in str(info.value)
    assert "register_sequence" in str(info.value)


def test_vendor_scope_exact_precedence_and_scanner_prefix_are_preserved(tmp_path):
    registry = SequenceRegistry(
        _write(
            tmp_path,
            {
                "flash": _entry(method_patterns=["exact:custom", "contains:family"]),
                "rare": _entry(
                    vendor="siemens",
                    method_patterns=["exact:family_specific", "contains:custom"],
                ),
            },
        )
    )
    assert registry.get_sequence_class(
        r"%scanner%\family_specific", "SIEMENS"
    ) is _class(_EXPECTED["rare"][0])
    assert registry.get_sequence_class('"%scanner%/custom"', "siemens") is _class(
        _EXPECTED["flash"][0]
    )
    assert registry.get_sequence_class("family_specific", "bruker") is _class(
        _EXPECTED["flash"][0]
    )
    assert registry.get_sequence_class("invented_unknown", "siemens") is None


def test_runtime_sequence_registration_keeps_direct_python_types(tmp_path):
    registry = SequenceRegistry(_write(tmp_path, {"flash": _entry()}))
    cls = _class(_EXPECTED["csi"][0])
    registry.register_sequence(
        "invented", cls, SequenceCategory.SPECTROSCOPY, ["exact:custom"], "siemens"
    )
    assert registry.get_sequence_class("custom", "siemens") is cls
    assert registry.get_sequence_class("custom", "bruker") is None


def test_unknown_classless_sequence_fails_with_file_and_remedy(tmp_path):
    path = _write(tmp_path, {"invented": _entry()})
    with pytest.raises(
        ConfigurationError, match="invented.*Python sequence mapping"
    ) as info:
        SequenceRegistry(path)
    assert info.value.config_path == str(path)
    assert "register_sequence" in str(info.value)


@pytest.mark.parametrize("field", ["category", "method_patterns"])
def test_classless_settings_keep_required_fields(tmp_path, field):
    entry = _entry()
    del entry[field]
    with pytest.raises(ConfigurationError, match=f"missing fields.*{field}"):
        SequenceRegistry(_write(tmp_path, {"flash": entry}))


def test_unknown_sequence_fields_and_explicit_null_class_stay_invalid(tmp_path):
    for entry, message in (
        (_entry(clas="typo"), "unknown fields.*clas"),
        (_entry(**{"class": None}), "has a 'class' field"),
    ):
        with pytest.raises(ConfigurationError, match=message):
            SequenceRegistry(_write(tmp_path, {"flash": entry}))


def test_repeated_external_sequence_key_is_rejected(tmp_path):
    path = tmp_path / "sequences.yaml"
    path.write_text("sequences:\n  flash: {}\n  flash: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate key 'flash'"):
        SequenceRegistry(path)


def test_python_sequence_import_failure_keeps_configuration_error_cause(
    tmp_path, monkeypatch
):
    path = _write(tmp_path, {"flash": _entry()})
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "_sequence_types":
            raise ImportError("invented missing sequence dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(
        ConfigurationError, match="Python sequence mapping cannot be imported"
    ) as info:
        SequenceRegistry(path)
    assert isinstance(info.value.__cause__, ImportError)
    assert info.value.config_path == str(path)


def test_classless_registries_own_their_patterns_and_do_not_construct_sequences(
    monkeypatch,
):
    cls = _class(_EXPECTED["flash"][0])

    def fail(*args, **kwargs):
        raise AssertionError("sequence construction must remain lazy")

    monkeypatch.setattr(cls, "__init__", fail)
    first, second = SequenceRegistry(), SequenceRegistry()
    first.get_sequence_info("flash")["method_patterns"].append("exact:invented")
    assert "exact:invented" not in second.get_sequence_info("flash")["method_patterns"]
    assert second.get_sequence_class("<Bruker:FLASH>") is cls
