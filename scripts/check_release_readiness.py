#!/usr/bin/env python3
"""Validate explicit release decisions before a package upload is allowed."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / ".github" / "release-readiness.json"
SUPPORT_POLICY_PATH = (
    PROJECT_ROOT / "src" / "hypermri" / "config" / "support_policy.json"
)
REAL_DATA_MANIFEST_PATH = (
    PROJECT_ROOT / "src" / "hypermri" / "datasets" / "registry.json"
)
SEMVER_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class Finding:
    """One pending decision or inconsistent release-policy value."""

    code: str
    message: str
    pending: bool = False

    def format(self) -> str:
        """Return a concise command-line representation."""
        state = "pending" if self.pending else "inconsistent"
        return f"[{state}:{self.code}] {self.message}"


def _load_json(path: Path) -> dict:
    """Load a JSON object and report malformed policy files clearly."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _decision_text(
    findings: list[Finding],
    policy: dict,
    key: str,
    description: str,
) -> str | None:
    """Require non-blank text before checking a recorded release decision."""
    value = policy.get(key)
    if value is None or value == "":
        findings.append(Finding(key, description, pending=True))
        return None
    if not isinstance(value, str) or not value.strip():
        findings.append(Finding(key, f"{key} must be a non-blank JSON string."))
        return None
    return value


def _stable_sequence_ids(findings: list[Finding], policy: dict) -> list[str] | None:
    """Validate the explicit stable Bruker raw-reconstruction sequence list."""
    key = "stable_bruker_raw_sequence_ids"
    value = policy.get(key)
    if value is None or value == []:
        findings.append(
            Finding(
                key, "approve at least one stable Bruker raw sequence", pending=True
            )
        )
        return None
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        findings.append(
            Finding(key, f"{key} must be a JSON list of unique, non-blank strings.")
        )
        return None
    return value


def _approval_value(
    findings: list[Finding],
    policy: dict,
    key: str,
    description: str,
) -> bool:
    """Require a JSON boolean while preserving missing/false pending decisions."""
    value = policy.get(key)
    if key not in policy or value is False:
        findings.append(Finding(key, description, pending=True))
    elif value is not True:
        findings.append(
            Finding(
                key,
                f"{key} must be a JSON boolean (true or false), "
                f"got {type(value).__name__}.",
            )
        )
    return value is True


def _historical_contributor_entries(markdown: str) -> list[str]:
    """Return reviewed roster entries outside fenced examples."""
    heading = "## Historical contributor roster"
    if heading not in markdown:
        return []
    section = markdown.split(heading, 1)[1]
    if "\n## " in section:
        section = section.split("\n## ", 1)[0]

    entries: list[str] = []
    in_fence = False
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("- ") or " — " not in line:
            continue
        public_name, contribution = line[2:].split(" — ", 1)
        if public_name.strip() and contribution.strip():
            entries.append(line)
    return entries


def validate_release_readiness(repository_root: Path = PROJECT_ROOT) -> list[Finding]:
    """Return pending and inconsistent release-readiness findings.

    Parameters
    ----------
    repository_root : pathlib.Path, optional
        Repository containing the release policy, packaged support and dataset
        registries, project metadata, contributor roster and release documents.
        Relative paths are resolved against the current working directory.

    Returns
    -------
    list of Finding
        Pending decisions and policy inconsistencies in deterministic check
        order. An empty list means all checked decisions are recorded and
        consistent. Only JSON ``true`` approves each boolean decision; missing
        fields or ``false`` are pending, while other types are inconsistent.
        Text decisions require non-blank strings and the stable sequence list
        requires unique non-blank strings. Missing, null or empty text/list decisions
        remain pending. Malformed policy files, including duplicate JSON keys
        at any nesting level, produce a ``policy-file`` finding. Project metadata
        is parsed once as TOML, accepting equivalent TOML syntax. Invalid TOML,
        including duplicate keys, or an absent/non-table ``project`` value
        produces a blocking ``project-file`` finding in both CLI modes.

    Raises
    ------
    OSError
        If required project metadata or release documentation cannot be read.
    UnicodeError
        If a required file cannot be decoded as UTF-8.

    Notes
    -----
    Validation reads local files eagerly without mutation, network access,
    subprocesses or cached results. It does not grant approval or change the
    publishing configuration.
    """
    root = repository_root.resolve()
    findings: list[Finding] = []
    try:
        policy = _load_json(root / ".github" / "release-readiness.json")
        support_policy = _load_json(
            root / "src" / "hypermri" / "config" / "support_policy.json"
        )
        real_data_manifest = _load_json(
            root / "src" / "hypermri" / "datasets" / "registry.json"
        )
    except ValueError as exc:
        return [Finding("policy-file", str(exc))]

    project_path = root / "pyproject.toml"
    try:
        project = tomllib.loads(project_path.read_text(encoding="utf-8")).get("project")
    except tomllib.TOMLDecodeError as exc:
        return [Finding("project-file", f"Could not parse {project_path}: {exc}")]
    if not isinstance(project, dict):
        return [
            Finding("project-file", f"{project_path} must define a [project] table.")
        ]

    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        findings.append(
            Finding("schema-version", "release policy schema_version must be 1")
        )

    release_target = _decision_text(
        findings, policy, "release_target", "select the final release target"
    )
    if isinstance(release_target, str) and not SEMVER_RE.fullmatch(release_target):
        findings.append(
            Finding("release-target", "release_target must be a semantic version")
        )

    distribution_name = _decision_text(
        findings,
        policy,
        "distribution_name",
        "select and reserve the PyPI distribution name",
    )
    project_name = project.get("name")
    if isinstance(distribution_name, str):
        if project_name != distribution_name:
            findings.append(
                Finding(
                    "distribution-name-mismatch",
                    "pyproject.toml project.name does not match the approved "
                    "distribution_name",
                )
            )

    license_identifier = _decision_text(
        findings,
        policy,
        "license_identifier",
        "approve a release license identifier",
    )
    project_license = project.get("license")
    if isinstance(license_identifier, str) and project_license != license_identifier:
        findings.append(
            Finding(
                "license-mismatch",
                "pyproject.toml project.license does not match license_identifier",
            )
        )

    _approval_value(
        findings,
        policy,
        "rights_review_complete",
        "complete the copyright and contribution-rights review",
    )

    contributors_attribution_complete = _approval_value(
        findings,
        policy,
        "contributors_attribution_complete",
        "review and publish the historical contributor roster",
    )
    contributors_text = (root / "CONTRIBUTORS.md").read_text(encoding="utf-8")
    if contributors_attribution_complete is True and not (
        _historical_contributor_entries(contributors_text)
    ):
        findings.append(
            Finding(
                "contributors-attribution-incomplete",
                "CONTRIBUTORS.md has no reviewed 'name — contribution' roster "
                "entry outside its fenced example",
            )
        )

    stable_ids = _stable_sequence_ids(findings, policy)
    configured_stable_ids = support_policy.get("bruker", {}).get(
        "stable_raw_reconstruction_sequence_ids"
    )
    if isinstance(stable_ids, list) and stable_ids != configured_stable_ids:
        findings.append(
            Finding(
                "stable-bruker-allowlist-mismatch",
                "release policy and support_policy.json list different stable "
                "Bruker raw sequences",
            )
        )

    reference_dataset_id = _decision_text(
        findings,
        policy,
        "reference_dataset_id",
        "select the public numerical reference dataset",
    )
    if isinstance(reference_dataset_id, str):
        reference_dataset = real_data_manifest.get("reference_datasets", {}).get(
            reference_dataset_id
        )
        if not isinstance(reference_dataset, dict):
            findings.append(
                Finding(
                    "reference-dataset-missing",
                    "reference_dataset_id is not present in "
                    "src/hypermri/datasets/registry.json",
                )
            )
        else:
            if (
                reference_dataset.get("release_status") != "released"
                or reference_dataset.get("enabled") is not True
            ):
                findings.append(
                    Finding(
                        "reference-dataset-disabled",
                        "the selected reference dataset must be released and enabled",
                    )
                )
            checksum = str(reference_dataset.get("checksum", "")).lower()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum):
                findings.append(
                    Finding(
                        "reference-dataset-checksum",
                        "the selected reference dataset must have a pinned SHA-256",
                    )
                )
            archive_size = reference_dataset.get("archive_size_bytes")
            archive_url = str(reference_dataset.get("archive_url", ""))
            if (
                not isinstance(archive_size, int)
                or archive_size <= 0
                or not archive_url.startswith("https://")
            ):
                findings.append(
                    Finding(
                        "reference-dataset-source",
                        "the selected reference dataset must pin an HTTPS archive "
                        "and its byte size",
                    )
                )
            required_attribution = (
                "creator",
                "dataset_version",
                "doi",
                "license",
                "license_url",
                "publisher",
                "source_url",
                "title",
            )
            if any(not reference_dataset.get(key) for key in required_attribution):
                findings.append(
                    Finding(
                        "reference-dataset-attribution",
                        "the selected reference dataset is missing citation or "
                        "license metadata",
                    )
                )
            stable_raw_cases = reference_dataset.get("stable_raw_cases")
            if isinstance(stable_ids, list) and (
                not isinstance(stable_raw_cases, dict)
                or sorted(stable_raw_cases) != sorted(stable_ids)
            ):
                findings.append(
                    Finding(
                        "reference-dataset-coverage",
                        "the selected reference dataset must map exactly one test "
                        "case for every stable Bruker raw sequence",
                    )
                )

    _approval_value(
        findings,
        policy,
        "reference_dataset_validation_complete",
        "add and scientifically approve numerical tests against the public "
        "reference dataset",
    )

    removal_release = _decision_text(
        findings,
        policy,
        "deprecation_removal_release",
        "select the deprecated-API removal release",
    )
    public_api_text = (root / "docs" / "source" / "api" / "public.rst").read_text(
        encoding="utf-8"
    )
    if isinstance(removal_release, str):
        if not SEMVER_RE.fullmatch(removal_release):
            findings.append(
                Finding(
                    "deprecation-removal-release",
                    "deprecation_removal_release must be a semantic version",
                )
            )
        if (
            removal_release not in public_api_text
            or "To be announced" in public_api_text
        ):
            findings.append(
                Finding(
                    "deprecation-documentation-mismatch",
                    "stable API documentation does not consistently name the "
                    "approved removal release",
                )
            )

    release_approver = _decision_text(
        findings,
        policy,
        "release_approver",
        "name the person authorised to approve releases",
    )
    maintainers_text = (root / "MAINTAINERS.md").read_text(encoding="utf-8")
    if isinstance(release_approver, str) and release_approver not in maintainers_text:
        findings.append(
            Finding(
                "release-approver-mismatch",
                "MAINTAINERS.md does not name the approved release approver",
            )
        )

    _approval_value(
        findings,
        policy,
        "public_snapshot_approved",
        "approve the final exported one-commit public snapshot",
    )

    return findings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate release decisions and their repository representation."
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="check consistency while allowing explicitly pending approvals",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release-readiness check."""
    args = build_parser().parse_args(argv)
    findings = validate_release_readiness()
    blocking = [
        finding for finding in findings if not (args.allow_pending and finding.pending)
    ]
    if blocking:
        print(f"Release readiness failed with {len(blocking)} finding(s):")
        for finding in blocking:
            print(f"- {finding.format()}")
        return 1

    pending_count = sum(finding.pending for finding in findings)
    if pending_count:
        print(
            "Release policy is internally consistent; "
            f"{pending_count} approval(s) remain pending."
        )
    else:
        print("Release readiness passed; all approvals are recorded and consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
