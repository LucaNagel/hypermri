"""Tests for the explicit release-readiness gate."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_readiness_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "check_release_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypermri_check_release_readiness", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


readiness = _load_readiness_script()
_APPROVAL_FIELDS = (
    "rights_review_complete",
    "contributors_attribution_complete",
    "reference_dataset_validation_complete",
    "public_snapshot_approved",
)
_TEXT_DECISIONS = (
    "release_target",
    "distribution_name",
    "license_identifier",
    "reference_dataset_id",
    "deprecation_removal_release",
    "release_approver",
)


@pytest.fixture
def _approved_policy(monkeypatch):
    """Provide valid approvals while retaining all other repository checks."""
    load_json = readiness._load_json
    policy = load_json(readiness.POLICY_PATH)
    policy.update(dict.fromkeys(_APPROVAL_FIELDS, True))

    def load_policy(path):
        if path == readiness.POLICY_PATH:
            return policy
        return load_json(path)

    monkeypatch.setattr(readiness, "_load_json", load_policy)
    return policy


@pytest.fixture
def _project_metadata_file(tmp_path, monkeypatch, _approved_policy):
    """Exercise project metadata in a complete, approved temporary repository."""
    for relative in (
        ".github/release-readiness.json",
        "src/hypermri/config/support_policy.json",
        "src/hypermri/datasets/registry.json",
        "pyproject.toml",
        "CONTRIBUTORS.md",
        "docs/source/api/public.rst",
        "MAINTAINERS.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readiness.PROJECT_ROOT / relative, destination)
    (tmp_path / ".github/release-readiness.json").write_text(
        json.dumps(_approved_policy), encoding="utf-8"
    )
    validate = readiness.validate_release_readiness
    monkeypatch.setattr(
        readiness, "validate_release_readiness", lambda: validate(tmp_path)
    )
    return tmp_path / "pyproject.toml"


@pytest.mark.parametrize(
    "contents",
    [
        "[project]\nname='hypermri'\nlicense='AGPL-3.0-only'\n",
        '[project]\nname = "hypermri" # package\nlicense = "AGPL-3.0-only" # SPDX\n',
        '["project"]\nname = "hypermri"\nlicense = "AGPL-3.0-only"\n',
        'project.name = "hypermri"\nproject.license = "AGPL-3.0-only"\n',
        '[project]\nname = """hypermri"""\nlicense = """AGPL-3.0-only"""\n',
    ],
    ids=["compact", "comments", "quoted-table", "dotted-keys", "multiline"],
)
def test_equivalent_toml_metadata_passes_both_cli_modes(
    _project_metadata_file, contents, capsys
):
    """TOML syntax must not change the approved project name or licence."""
    _project_metadata_file.write_text(contents, encoding="utf-8")

    assert readiness.validate_release_readiness() == []
    assert readiness.main([]) == 0
    assert readiness.main(["--allow-pending"]) == 0
    capsys.readouterr()


@pytest.mark.parametrize(
    "contents",
    [
        '[project]\nname = "hypermri"\nname = "other"\nlicense = "AGPL-3.0-only"\n',
        '[project]\nname = "hypermri"\nlicense = "AGPL-3.0-only"\n[project]\nname = "other"\n',
        '[project]\nname = "hypermri"\nlicense = "AGPL-3.0-only"\n[tool.test]\nvalue = [\n',
        'project = "not a table"\n',
        '[[project]]\nname = "hypermri"\nlicense = "AGPL-3.0-only"\n',
        '[tool.test]\nname = "hypermri"\nlicense = "AGPL-3.0-only"\n',
    ],
    ids=[
        "duplicate-key",
        "duplicate-table",
        "malformed-other-table",
        "scalar-project",
        "array-project",
        "missing-project",
    ],
)
def test_invalid_project_toml_blocks_both_cli_modes(
    _project_metadata_file, contents, capsys
):
    """Invalid metadata cannot authorize release or pass consistency mode."""
    _project_metadata_file.write_text(contents, encoding="utf-8")

    findings = readiness.validate_release_readiness()
    assert len(findings) == 1
    assert findings[0].code == "project-file"
    assert findings[0].pending is False
    assert "pyproject.toml" in findings[0].message
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert "[inconsistent:project-file]" in capsys.readouterr().out


def test_boolean_true_approvals_pass_readiness_and_both_cli_modes(
    _approved_policy, capsys
):
    assert readiness.validate_release_readiness() == []
    assert readiness.main([]) == 0
    assert readiness.main(["--allow-pending"]) == 0
    assert "all approvals are recorded and consistent" in capsys.readouterr().out


@pytest.mark.parametrize("field", _APPROVAL_FIELDS)
@pytest.mark.parametrize("state", ["missing", False], ids=["missing", "false"])
def test_missing_or_false_approval_remains_pending(
    _approved_policy, field, state, capsys
):
    if state == "missing":
        _approved_policy.pop(field)
    else:
        _approved_policy[field] = state

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is True
    assert readiness.main([]) == 1
    assert f"[pending:{field}]" in capsys.readouterr().out
    assert readiness.main(["--allow-pending"]) == 0
    assert "1 approval(s) remain pending" in capsys.readouterr().out


@pytest.mark.parametrize("field", _APPROVAL_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param("false", id="string-false"),
        pytest.param("true", id="string-true"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param([], id="empty-list"),
        pytest.param([True], id="list"),
        pytest.param({}, id="empty-object"),
        pytest.param({"approved": True}, id="object"),
    ],
)
def test_invalid_approval_type_blocks_readiness_and_both_cli_modes(
    _approved_policy, field, value, capsys
):
    _approved_policy[field] = value

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is False
    assert field in findings[0].message
    assert "JSON boolean" in findings[0].message
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert f"[inconsistent:{field}]" in capsys.readouterr().out


@pytest.mark.parametrize("field", _TEXT_DECISIONS)
@pytest.mark.parametrize("value", [False, True, 7, 1.5, [], ["value"], {}, " \t"])
def test_invalid_text_decision_blocks_both_cli_modes(
    _approved_policy, field, value, capsys
):
    _approved_policy[field] = value

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is False
    assert field in findings[0].message
    assert "non-blank JSON string" in findings[0].message
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert f"[inconsistent:{field}]" in capsys.readouterr().out


@pytest.mark.parametrize("field", _TEXT_DECISIONS)
@pytest.mark.parametrize("state", ["missing", None, ""])
def test_unset_text_decision_remains_pending(_approved_policy, field, state, capsys):
    if state == "missing":
        _approved_policy.pop(field)
    else:
        _approved_policy[field] = state

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is True
    assert readiness.main([]) == 1
    assert readiness.main(["--allow-pending"]) == 0
    capsys.readouterr()


@pytest.mark.parametrize(
    "value",
    [False, True, 7, "flash", {}, ["flash", 7], [""], [" "], ["flash", "flash"]],
)
def test_invalid_sequence_decision_blocks_both_cli_modes(
    _approved_policy, value, capsys
):
    field = "stable_bruker_raw_sequence_ids"
    _approved_policy[field] = value

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is False
    assert field in findings[0].message
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert f"[inconsistent:{field}]" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["missing", None, []])
def test_unset_sequence_decision_remains_pending(_approved_policy, state, capsys):
    field = "stable_bruker_raw_sequence_ids"
    if state == "missing":
        _approved_policy.pop(field)
    else:
        _approved_policy[field] = state

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == field
    assert findings[0].pending is True
    assert readiness.main([]) == 1
    assert readiness.main(["--allow-pending"]) == 0
    capsys.readouterr()


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_schema_version_requires_integer_one(_approved_policy, version, capsys):
    _approved_policy["schema_version"] = version

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == "schema-version"
    assert findings[0].pending is False
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert "[inconsistent:schema-version]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "contents, key",
    [
        (
            '{"rights_review_complete": false, "rights_review_complete": true}',
            "rights_review_complete",
        ),
        (
            '{"reference_datasets": {"example": {"enabled": false, "enabled": true}}}',
            "enabled",
        ),
        ('{"schema_version": 1, "schema_versi\\u006fn": 1}', "schema_version"),
    ],
)
def test_duplicate_json_keys_block_both_cli_modes(
    tmp_path, monkeypatch, contents, key, capsys
):
    policy_path = tmp_path / "release-readiness.json"
    policy_path.write_text(contents, encoding="utf-8")
    load_json = readiness._load_json

    def load_policy(path):
        return load_json(policy_path if path == readiness.POLICY_PATH else path)

    monkeypatch.setattr(readiness, "_load_json", load_policy)

    findings = readiness.validate_release_readiness()

    assert len(findings) == 1
    assert findings[0].code == "policy-file"
    assert findings[0].pending is False
    assert "Duplicate JSON key" in findings[0].message
    assert key in findings[0].message
    assert str(policy_path) in findings[0].message
    for arguments in ([], ["--allow-pending"]):
        assert readiness.main(arguments) == 1
        assert "[inconsistent:policy-file]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "version",
    ["01.0.0", "0.1.0rc1", "0.1.0foo", "0.1", "v0.1.0"],
)
def test_release_target_rejects_noncanonical_final_versions(version: str):
    assert readiness.SEMVER_RE.fullmatch(version) is None


def test_current_release_policy_records_every_approval(capsys):
    assert readiness.validate_release_readiness() == []
    assert readiness.main([]) == 0
    assert "all approvals are recorded" in capsys.readouterr().out


def test_strict_cli_blocks_pending_release(_approved_policy, capsys):
    _approved_policy["reference_dataset_validation_complete"] = False

    assert readiness.main([]) == 1
    output = capsys.readouterr().out
    assert "public reference dataset" in output
    assert "select and reserve the PyPI distribution name" not in output


def test_consistency_cli_allows_recorded_pending_decisions(_approved_policy, capsys):
    _approved_policy["public_snapshot_approved"] = False

    assert readiness.main(["--allow-pending"]) == 0
    assert "internally consistent" in capsys.readouterr().out


def test_contributor_roster_parser_rejects_empty_or_example_only_sections():
    empty = """\
## Historical contributor roster

```text
- Preferred public name — contribution area.
```

## What this list means
"""

    assert readiness._historical_contributor_entries(empty) == []


def test_contributor_roster_parser_accepts_reviewed_public_entry():
    roster = """\
## Historical contributor roster

- Ada Example ([ORCID](https://orcid.org/0000-0000-0000-0000)) — Parser and tests.

## What this list means
"""

    assert readiness._historical_contributor_entries(roster) == [
        "- Ada Example ([ORCID](https://orcid.org/0000-0000-0000-0000)) — Parser and tests."
    ]


def test_current_contributor_roster_uses_reviewable_entries():
    contributors_path = Path(__file__).resolve().parents[2] / "CONTRIBUTORS.md"

    entries = readiness._historical_contributor_entries(
        contributors_path.read_text(encoding="utf-8")
    )

    assert len(entries) >= 5
