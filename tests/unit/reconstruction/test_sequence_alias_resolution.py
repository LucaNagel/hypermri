"""Characterize how raw method names resolve to canonical sequence IDs.

``method_name_to_sequence_id`` is the lookup the reconstruction selection and
the scanner-image reconstruction use. These tests recorded the results of the
former ``resolve_sequence_alias`` before the lookup moved into
``hypermri.sequence_mapping``; the moved lookup must keep them.
The exact aliases come from the packaged ``sequence_mapping.yaml``; the
``startswith:`` case uses a one-entry mapping written for the test, because
no packaged entry has such a pattern.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import hypermri.sequence_mapping as sequence_mapping_module
import hypermri.sequences.registry as registry_module
from hypermri.sequence_mapping import load_sequence_mapping, method_name_to_sequence_id

_PACKAGED_MAPPING = (
    Path(registry_module.__file__).parent / "config" / "sequence_mapping.yaml"
)


def _packaged_sequences() -> dict:
    with _PACKAGED_MAPPING.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["sequences"]


def _packaged_exact_aliases() -> list[tuple[str, str]]:
    return [
        (pattern[len("exact:") :], canonical)
        for canonical, entry in _packaged_sequences().items()
        for pattern in entry["method_patterns"]
        if pattern.lower().startswith("exact:")
    ]


@pytest.mark.parametrize(("alias", "canonical"), _packaged_exact_aliases())
def test_every_packaged_exact_alias_resolves_to_its_sequence(alias, canonical):
    assert method_name_to_sequence_id(alias) == canonical
    assert method_name_to_sequence_id(alias.upper()) == canonical


@pytest.mark.parametrize("canonical", list(_packaged_sequences()))
def test_every_canonical_id_resolves_to_itself(canonical):
    assert method_name_to_sequence_id(canonical) == canonical


def test_spin_echo_id_resolves_although_no_pattern_names_it():
    # The registry lookup alone finds no pattern for "spin_echo"; the
    # canonical-ID step of the alias lookup is what resolves it.
    assert registry_module.get_sequence_registry().resolve_sequence("spin_echo") is None
    assert method_name_to_sequence_id("spin_echo") == "spin_echo"


def test_contains_pattern_resolves_an_unlisted_method_name():
    assert method_name_to_sequence_id("<User:my_flash_variant>") == "flash"


def test_startswith_pattern_resolves_through_the_mapping(tmp_path, monkeypatch):
    mapping = tmp_path / "sequence_mapping.yaml"
    mapping.write_text(
        "sequences:\n"
        "  flash:\n"
        "    category: anatomical\n"
        "    method_patterns:\n"
        '      - "startswith:<lab:"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sequence_mapping_module, "_packaged_entries", load_sequence_mapping(mapping)
    )

    assert method_name_to_sequence_id("<Lab:anything>") == "flash"
    assert method_name_to_sequence_id("anything <lab:") is None


@pytest.mark.parametrize("name", ["unregistered_method", ""])
def test_unknown_method_name_resolves_to_none(name):
    assert method_name_to_sequence_id(name) is None


def test_none_resolves_to_none():
    assert method_name_to_sequence_id(None) is None


def test_sequence_mapping_imports_no_sequence_class(tmp_path, package_subprocess_env):
    # Reconstruction imports this module at module level; it must stay below
    # the ``sequences`` package in the dependency direction.
    source = (
        "import sys\n"
        "from hypermri.sequence_mapping import method_name_to_sequence_id\n"
        "assert method_name_to_sequence_id('<Bruker:FLASH>') == 'flash'\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('hypermri.sequences'))\n"
        "assert not loaded, loaded\n"
    )
    env = package_subprocess_env.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
