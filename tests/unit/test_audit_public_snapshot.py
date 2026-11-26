"""Tests for the read-only public snapshot audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest


def _load_audit_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "audit_public_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypermri_audit_public_snapshot", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_script()

APPROVED_SYNTHETIC_FIXTURES = (
    "tests/fixtures/synthetic_csi.py",
    "tests/fixtures/synthetic_metabolite_detection.py",
    "tests/fixtures/synthetic_noise_quantification.py",
    "tests/fixtures/synthetic_spectroscopy_uncertainty.py",
    "tests/fixtures/synthetic_t1.py",
)


def _codes(findings) -> set[str]:
    return {finding.code for finding in findings}


@pytest.mark.parametrize(
    "path",
    [
        "private_scan/2dseq",
        "private_scan/rawdata.job0",
        "private_scan.fid/procpar",
        "tests/fixtures/reference.json",
        "tests/data/payload.txt",
        "tests/reference.npy",
        "scanner_export.dat",
        "scanner_export.nii.gz",
        "scanner_export.zip",
    ],
)
def test_path_audit_rejects_raw_scanner_and_test_data(path: str):
    findings = audit.find_path_violations(path)

    assert _codes(findings) & {"raw-or-data-file", "test-data-file"}


def test_path_audit_rejects_agent_instructions():
    assert _codes(audit.find_path_violations("AGENTS.md")) == {
        "private-development-file"
    }
    assert audit.find_path_violations("docs/AGENTS.md") == []


@pytest.mark.parametrize("path", APPROVED_SYNTHETIC_FIXTURES)
def test_path_audit_allows_only_named_synthetic_python_fixtures(path: str):
    assert audit.find_path_violations(path) == []


@pytest.mark.parametrize(
    "path, expected_codes",
    [
        ("tests/fixtures/unapproved_synthetic.py", {"test-data-file"}),
        (
            "tests/fixtures/fid",
            {"raw-or-data-file", "test-data-file"},
        ),
    ],
)
def test_path_audit_still_rejects_unapproved_fixture_files(path, expected_codes):
    assert expected_codes <= _codes(audit.find_path_violations(path))


def test_allowed_synthetic_fixture_still_checks_text_for_unapproved_email(
    tmp_path: Path,
):
    relative_path = APPROVED_SYNTHETIC_FIXTURES[-1]
    fixture = tmp_path / relative_path
    fixture.parent.mkdir(parents=True)
    private_address = "former.contributor" + "@" + "private.example"
    fixture.write_text(private_address, encoding="utf-8")

    findings = audit.audit_snapshot(tmp_path, [relative_path])

    assert _codes(findings) == {"unapproved-email"}


def test_path_audit_allows_only_approved_documentation_data_paths():
    approved_paths = [
        "docs/source/_static/media/example/plot.png",
        "docs/source/_static/media/example/video.mp4",
        "docs/source/_static/media/example/video.mov",
        "docs/source/_static/media/README.md",
    ]

    assert all(
        audit.find_path_violations(approved_path) == []
        for approved_path in approved_paths
    )
    assert "unapproved-notebook" in _codes(
        audit.find_path_violations(
            "docs/source/tutorials/notebooks/unapproved_tutorial.ipynb"
        )
    )


@pytest.mark.parametrize(
    "name",
    [
        "tutorial_csi_synthetic.ipynb",
        "tutorial_fit_and_overlay.ipynb",
        "tutorial_hp_pyruvate_csi_static.ipynb",
        "tutorial_statistical_spectroscopy_fitting.ipynb",
    ],
)
def test_retired_notebooks_no_longer_have_snapshot_approval(name):
    path = f"docs/source/tutorials/notebooks/{name}"
    assert "unapproved-notebook" in _codes(audit.find_path_violations(path))


@pytest.mark.parametrize(
    "path",
    [
        "docs/source/_static/media/example/scanner_export.dcm",
        "docs/source/_static/media/example/dataset.zip",
        "docs/source/_static/media/example/scan/fid",
        "docs/source/_static/media/example/scan/2dseq",
        "docs/source/_static/media/example/scan/rawdata.job0",
    ],
)
def test_path_audit_rejects_raw_data_inside_documentation_media(path: str):
    codes = _codes(audit.find_path_violations(path))

    assert "unapproved-documentation-media" in codes
    assert "raw-or-data-file" in codes


def test_path_audit_rejects_unapproved_documentation_media_type():
    findings = audit.find_path_violations("docs/source/_static/media/example/notes.txt")

    assert _codes(findings) == {"unapproved-documentation-media"}


@pytest.mark.parametrize(
    "path, expected_code",
    [
        (".env", "private-environment-file"),
        (".env.local", "private-environment-file"),
        (".pypirc", "private-environment-file"),
        ("deployment/credentials.json", "private-environment-file"),
        ("keys/id_ed25519", "private-key-file"),
        ("keys/signing.key", "private-key-file"),
        ("keys/signing.p12", "private-key-file"),
    ],
)
def test_path_audit_rejects_private_configuration_and_key_files(
    path: str,
    expected_code: str,
):
    assert expected_code in _codes(audit.find_path_violations(path))


@pytest.mark.parametrize(
    "path",
    [".env.example", ".env-template", "config/sample.env", "certificates/public.crt"],
)
def test_path_audit_allows_environment_templates_and_public_certificates(path: str):
    assert audit.find_path_violations(path) == []


def test_text_audit_allows_approved_project_addresses_and_git_transport():
    allowed_text = "\n".join(
        [
            "luca.nagel@protonmail.com",
            "luca.nagel@posteo.de",
            "noreply@hypermri.invalid",
            "git@github.com:example/project.git",
        ]
    )
    private_address = "former.contributor" + "@" + "private.example"

    assert audit.find_text_violations("SECURITY.md", allowed_text) == []
    assert "unapproved-email" in _codes(
        audit.find_text_violations("AUTHORS.txt", private_address)
    )


def test_text_audit_rejects_old_repository_link_without_embedding_it_in_test():
    old_url = "https://github.com/" + "schillinglab/" + "hypermri"

    findings = audit.find_text_violations("README.md", old_url)

    assert _codes(findings) == {"old-repository-link"}


@pytest.mark.parametrize(
    "personal_path",
    [
        "/" + "Users" + "/" + "alice" + "/project",
        "/" + "home" + "/" + "researcher" + "/project",
        "C:" + "\\" + "Users" + "\\" + "Alice" + "\\" + "project",
        "/" + "mnt" + "/" + "c" + "/" + "Users" + "/" + "alice" + "/project",
    ],
)
def test_text_audit_rejects_personal_absolute_paths(personal_path: str):
    assert "personal-absolute-path" in _codes(
        audit.find_text_violations("script.py", personal_path)
    )


@pytest.mark.parametrize(
    "portable_path",
    [
        "${PROJECT_ROOT}/data",
        "$HOME/project",
        "/home/user/project",
        "/Users/runner/project",
        "C:\\Users\\runneradmin\\project",
    ],
)
def test_text_audit_allows_portable_or_ci_paths(portable_path: str):
    assert audit.find_text_violations("docs/example.rst", portable_path) == []


def test_text_audit_rejects_private_key_marker():
    marker = "-----BEGIN " + "OPENSSH PRIVATE" + " KEY-----"

    assert "private-key-content" in _codes(
        audit.find_text_violations("notes.txt", marker)
    )


def test_binary_detection_does_not_decode_binary_payloads():
    assert audit.is_probably_binary(b"\x00\xff\x01")
    assert not audit.is_probably_binary(b"plain UTF-8 repository text\n")


def test_parse_git_output_deduplicates_and_sorts_paths():
    output = b"zeta.py\0README.md\0zeta.py\0"

    assert audit.parse_git_path_output(output) == [
        PurePosixPath("README.md"),
        PurePosixPath("zeta.py"),
    ]


def test_git_listing_includes_tracked_and_unignored_new_files(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=b"tracked.py\0new_file.py\0")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    paths = audit.list_snapshot_paths(Path("/repository"))

    assert paths == [PurePosixPath("new_file.py"), PurePosixPath("tracked.py")]
    assert calls == [
        (
            [
                "git",
                "-C",
                "/repository",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            {"check": True, "capture_output": True},
        )
    ]


def test_audit_snapshot_combines_path_and_text_findings(tmp_path: Path):
    raw_path = tmp_path / "scan.dat"
    raw_path.write_bytes(b"\x00raw scanner payload")
    text_path = tmp_path / "notes.txt"
    private_address = "former.contributor" + "@" + "private.example"
    text_path.write_text(private_address, encoding="utf-8")

    findings = audit.audit_snapshot(tmp_path, ["scan.dat", "notes.txt"])

    assert _codes(findings) == {"raw-or-data-file", "unapproved-email"}
