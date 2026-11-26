"""Exercise real-data availability using invented files and provenance only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)


def _preserve_python_path(monkeypatch):
    # Pytester's temporary home must not hide user-installed test dependencies.
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(p for p in sys.path if Path(p).is_absolute())
    )


@pytest.mark.parametrize("required", [False, True], ids=["optional", "required"])
def test_real_data_availability_modes(pytester, monkeypatch, required):
    """Missing or unverified data skip locally and fail in required CI mode."""
    _preserve_python_path(monkeypatch)
    data_root = pytester.path / "data"
    datasets = {}
    unavailable = (
        "unreleased",
        "disabled",
        "missing_directory",
        "missing_file",
        "lfs_pointer",
        "missing_marker",
        "malformed_marker",
        "mismatched_marker",
    )
    for name in (*unavailable, "valid"):
        dataset = {
            "relative_path": name,
            "enabled": name != "disabled",
            "release_status": "unreleased" if name == "unreleased" else "released",
            "archive_name": "invented.zip",
            "archive_size_bytes": 123,
            "checksum": "sha256:" + "a" * 64,
            "required_paths": ["payload.txt"],
        }
        datasets[name] = dataset
        if name == "missing_directory":
            continue
        path = data_root / name
        path.mkdir(parents=True)
        if name != "missing_file":
            payload = (
                "version https://git-lfs.github.com/spec/v1\n"
                if name == "lfs_pointer"
                else "invented data"
            )
            (path / "payload.txt").write_text(payload, encoding="utf-8")
        provenance = {
            "schema_version": 1,
            "dataset_id": name,
            "relative_path": name,
            "archive_name": dataset["archive_name"],
            "archive_size_bytes": dataset["archive_size_bytes"],
            "checksum": dataset["checksum"],
        }
        if name == "mismatched_marker":
            provenance["checksum"] = "sha256:" + "b" * 64
        if name != "missing_marker":
            marker = "{" if name == "malformed_marker" else json.dumps(provenance)
            (path / ".hypermri-test-data.json").write_text(marker, encoding="utf-8")

    manifest = {"data_root_env": "HYPERMRI_TEST_DATA", "datasets": datasets}
    fixture_source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(
        encoding="utf-8"
    )
    pytester.makeconftest(
        fixture_source
        + f"""

@pytest.fixture(scope="session")
def real_data_manifest():
    return {manifest!r}
"""
    )
    pytester.makepyfile(
        test_availability=(
            "import pytest\n\n"
            + f"@pytest.mark.parametrize('dataset', {list(datasets)!r})\n"
            + "def test_resolve(real_dataset_path, dataset):\n"
            + "    path = real_dataset_path(dataset)\n"
            + "    assert path.name == dataset\n"
        )
    )
    monkeypatch.setenv("HYPERMRI_TEST_DATA", str(data_root))
    arguments = ["-o", "addopts=", "-q", "--tb=short"]
    if required:
        arguments.append("--require-real-data")
    result = pytester.runpytest_subprocess(*arguments)
    if required:
        result.assert_outcomes(passed=1, failed=len(unavailable))
        assert result.ret == pytest.ExitCode.TESTS_FAILED
    else:
        result.assert_outcomes(passed=1, skipped=len(unavailable))
        assert result.ret == pytest.ExitCode.OK


def test_unknown_real_dataset_always_fails(pytester, monkeypatch):
    """A misspelled dataset ID must never be treated as optional absence."""
    _preserve_python_path(monkeypatch)
    fixture_source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(
        encoding="utf-8"
    )
    pytester.makeconftest(
        fixture_source
        + """

@pytest.fixture(scope="session")
def real_data_manifest():
    return {'datasets': {}}
"""
    )
    pytester.makepyfile(
        test_unknown=(
            "def test_resolve(real_dataset_path):\n    real_dataset_path('unknown')\n"
        )
    )
    result = pytester.runpytest_subprocess("-o", "addopts=", "-q", "--tb=short")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Unknown real-data dataset: unknown*"])
