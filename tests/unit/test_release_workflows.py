"""Protect cross-platform package validation in release workflows."""

from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
RELEASE_WORKFLOWS = ("publish.yml", "testpypi.yml")
REQUIRED_OS = {"ubuntu-latest", "macos-latest", "windows-latest"}


def _load_workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


def _matrix_rows(matrix: dict) -> list[dict]:
    assert not matrix.get("exclude")
    if "include" in matrix:
        return matrix["include"]

    keys = tuple(matrix)
    return [
        dict(zip(keys, values, strict=True)) for values in product(*matrix.values())
    ]


def _effective_shell(workflow: dict, job: dict, step: dict) -> str | None:
    return (
        step.get("shell")
        or job.get("defaults", {}).get("run", {}).get("shell")
        or workflow.get("defaults", {}).get("run", {}).get("shell")
    )


def _external_action_uses(value: object) -> list[str]:
    """Return external GitHub Action references from a parsed workflow."""
    uses: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uses" and isinstance(item, str) and not item.startswith("./"):
                uses.append(item)
            else:
                uses.extend(_external_action_uses(item))
    elif isinstance(value, list):
        for item in value:
            uses.extend(_external_action_uses(item))
    return uses


def test_all_external_workflow_actions_are_pinned_to_full_commit_shas():
    workflows = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])
    external_uses = [
        (path.name, reference)
        for path in workflows
        for reference in _external_action_uses(_load_workflow(path.name))
    ]

    assert workflows
    assert external_uses
    assert all(
        re.fullmatch(r"[^@]+@[0-9a-f]{40}", reference) for _, reference in external_uses
    ), external_uses


def test_dependabot_updates_only_github_actions_weekly_in_one_group():
    config_path = WORKFLOW_DIR.parent / "dependabot.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["version"] == 2
    assert config["updates"] == [
        {
            "package-ecosystem": "github-actions",
            "directory": "/",
            "schedule": {"interval": "weekly"},
            "groups": {"github-actions": {"patterns": ["*"]}},
        }
    ]


@pytest.mark.parametrize("workflow_name", RELEASE_WORKFLOWS)
def test_release_wheel_validation_is_required_on_all_supported_platforms(
    workflow_name: str,
):
    workflow = _load_workflow(workflow_name)
    job = workflow["jobs"]["test-wheel"]
    rows = _matrix_rows(job["strategy"]["matrix"])

    assert len(rows) == 3
    assert {row["os"] for row in rows} == REQUIRED_OS
    assert {row["python-version"] for row in rows} == {"3.12"}
    assert all(set(row) == {"os", "python-version"} for row in rows)
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["needs"] == "build"
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job.get("steps", []))
    assert all(
        _effective_shell(workflow, job, step) == "bash"
        for step in job.get("steps", [])
        if "run" in step
    )


@pytest.mark.parametrize("workflow_name", RELEASE_WORKFLOWS)
def test_release_workflow_builds_once_and_tests_and_publishes_same_artifact(
    workflow_name: str,
):
    workflow = _load_workflow(workflow_name)
    jobs = workflow["jobs"]
    build_calls = [
        (job_name, step)
        for job_name, job in jobs.items()
        for step in job.get("steps", [])
        if "python -m build" in step.get("run", "")
    ]

    assert len(build_calls) == 1
    assert build_calls[0][0] == "build"
    assert "quality" in jobs
    assert set(jobs["publish"]["needs"]) == {"build", "quality", "test-wheel"}

    upload_steps = [
        step
        for step in jobs["build"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(upload_steps) == 1
    artifact_name = upload_steps[0]["with"]["name"]

    for job_name in ("test-wheel", "publish"):
        download_steps = [
            step
            for step in jobs[job_name]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert len(download_steps) == 1
        assert download_steps[0]["with"]["name"] == artifact_name


@pytest.mark.parametrize("workflow_name", RELEASE_WORKFLOWS)
def test_release_reference_tests_validate_the_same_built_wheel(workflow_name: str):
    workflow = _load_workflow(workflow_name)
    jobs = workflow["jobs"]
    quality = jobs["quality"]

    assert quality["needs"] == "build"
    assert quality["uses"] == "./.github/workflows/release-quality.yml"
    assert "if" not in quality
    assert "continue-on-error" not in quality

    artifact_name = next(
        step["with"]["name"]
        for step in jobs["build"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    reference = _load_workflow("release-quality.yml")["jobs"]["reference-data"]
    steps = reference["steps"]
    download_index, download = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    install_index, install = next(
        (index, step)
        for index, step in enumerate(steps)
        if "pip', 'install'" in step.get("run", "")
    )
    assert download["with"] == {"name": artifact_name, "path": "dist/"}
    assert download_index < install_index
    assert "Path('dist').glob('*.whl')" in install["run"]
    assert "{wheel}[test]" in install["run"]
    assert "-e " not in install["run"]
    assert "pip check" in install["run"]
    assert "sysconfig.get_path('purelib')" in install["run"]
    assert "is_relative_to(expected)" in install["run"]

    test_index, test = next(
        (index, step)
        for index, step in enumerate(steps)
        if "--require-real-data" in step.get("run", "")
    )
    assert install_index < test_index
    assert "python -I -m pytest" in test["run"]
    assert "-m integration" in test["run"]
    assert "if" not in reference
    assert "continue-on-error" not in reference
    assert all("if" not in step and "continue-on-error" not in step for step in steps)


def test_source_installation_workflow_requires_all_supported_platforms():
    workflow = _load_workflow("install_test.yml")
    job = workflow["jobs"]["install-and-test"]
    rows = _matrix_rows(job["strategy"]["matrix"])

    assert len(rows) == 3
    assert {row["os"] for row in rows} == REQUIRED_OS
    assert {row["python-version"] for row in rows} == {"3.12"}
    assert all(set(row) == {"os", "python-version"} for row in rows)
    assert job["runs-on"] == "${{ matrix.os }}"
    assert "continue-on-error" not in job
