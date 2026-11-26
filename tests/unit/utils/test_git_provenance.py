"""Preserve Git provenance values and the existing unavailable-repository labels."""

import subprocess
from pathlib import Path

from hypermri.utils.system.environment import get_git_commit_hash
from hypermri.vendors import bruker, siemens

_ROOT = Path(__file__).resolve().parents[3]


def test_handler_and_environment_hashes_identify_the_checkout(monkeypatch):
    """All entry points return the SHA of their explicitly selected checkout."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(_ROOT)
    vendor_source = _ROOT / "src" / "hypermri" / "vendors"
    monkeypatch.setattr(bruker, "__file__", str(vendor_source / "bruker.py"))
    monkeypatch.setattr(siemens, "__file__", str(vendor_source / "siemens.py"))
    assert get_git_commit_hash() == expected
    assert (
        bruker.BrukerHandler.__new__(bruker.BrukerHandler)._get_git_commit_hash()
        == expected
    )
    assert (
        siemens.SiemensHandler.__new__(
            siemens.SiemensHandler
        )._SiemensHandler__get_git_commit_hash()
        == expected
    )


def test_handlers_do_not_claim_the_callers_git_history(monkeypatch, tmp_path):
    """An installed package outside Git must not claim its caller's checkout."""
    monkeypatch.chdir(_ROOT)
    monkeypatch.setattr(bruker, "__file__", str(tmp_path / "bruker.py"))
    monkeypatch.setattr(siemens, "__file__", str(tmp_path / "siemens.py"))

    assert get_git_commit_hash() != "unknown"
    assert (
        bruker.BrukerHandler.__new__(bruker.BrukerHandler)._get_git_commit_hash()
        == "Could not find a git repository."
    )
    assert (
        siemens.SiemensHandler.__new__(
            siemens.SiemensHandler
        )._SiemensHandler__get_git_commit_hash()
        == "Could not find a git repository."
    )


def test_existing_missing_repository_labels(monkeypatch, tmp_path):
    """A missing Git repository does not invent provenance or load scan samples."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bruker, "__file__", str(tmp_path / "invented.py"))
    monkeypatch.setattr(siemens, "__file__", str(tmp_path / "invented.py"))
    assert get_git_commit_hash() == "unknown"
    assert (
        bruker.BrukerHandler.__new__(bruker.BrukerHandler)._get_git_commit_hash()
        == "Could not find a git repository."
    )
    assert (
        siemens.SiemensHandler.__new__(
            siemens.SiemensHandler
        )._SiemensHandler__get_git_commit_hash()
        == "Could not find a git repository."
    )


def test_missing_git_executable_returns_the_explicit_label(monkeypatch):
    """A core-only installation can load metadata without Git tooling."""
    from hypermri.utils.system import environment

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("invented unavailable executable")

    monkeypatch.setattr(environment.subprocess, "run", unavailable)
    assert get_git_commit_hash() == "unknown"
    assert (
        get_git_commit_hash(missing="unavailable provenance")
        == "unavailable provenance"
    )
