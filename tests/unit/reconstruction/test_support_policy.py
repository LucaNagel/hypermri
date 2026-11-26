"""Structural tests for the machine-readable support policy."""

from __future__ import annotations

import json
from pathlib import Path

from hypermri.reconstruction.selection import RECONSTRUCTIONS
from hypermri.sequences.registry import SequenceRegistry


def _project_root() -> Path:
    return Path(__file__).parents[3]


def _support_policy() -> dict:
    policy_path = (
        _project_root() / "src" / "hypermri" / "config" / "support_policy.json"
    )
    return json.loads(policy_path.read_text(encoding="utf-8"))


def test_stable_bruker_raw_allowlist_is_explicit_and_unique():
    sequence_ids = _support_policy()["bruker"]["stable_raw_reconstruction_sequence_ids"]

    assert sequence_ids == sorted(sequence_ids)
    assert len(sequence_ids) == len(set(sequence_ids))
    assert sequence_ids == [
        "csi",
        "fair_rare",
        "flash",
        "isis",
        "nspect",
        "press",
        "rare",
        "rarevtr",
        "singlepulse",
        "steam",
    ]


def test_every_stable_bruker_raw_id_has_an_enabled_explicit_strategy():
    sequence_ids = _support_policy()["bruker"]["stable_raw_reconstruction_sequence_ids"]

    canonical_ids = {
        entry["sequence_name"] for entry in SequenceRegistry().list_sequences().values()
    }

    for sequence_id in sequence_ids:
        assert sequence_id in canonical_ids, sequence_id
        # An explicit entry, not the vendor default of find_reconstruction.
        entry = RECONSTRUCTIONS.get(("bruker", sequence_id, "raw"))
        assert entry is not None, sequence_id
        _strategy, options = entry
        assert options.get("no_reconstruction") is not True, sequence_id


def test_documented_allowlist_matches_machine_readable_policy():
    support_page = (_project_root() / "docs" / "source" / "support.rst").read_text(
        encoding="utf-8"
    )
    sequence_ids = _support_policy()["bruker"]["stable_raw_reconstruction_sequence_ids"]

    for sequence_id in sequence_ids:
        assert f"``{sequence_id}``" in support_page
