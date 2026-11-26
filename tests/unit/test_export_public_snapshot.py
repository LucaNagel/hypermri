"""Tests for the history-free public-snapshot exporter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_export_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "export_public_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypermri_export_public_snapshot", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_export_script()


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def source_repository(tmp_path: Path) -> Path:
    """Create a repository with a now-ignored file and agent instructions."""
    repository = tmp_path / "private"
    repository.mkdir()
    _git(repository, "init", "--quiet")

    (repository / "kept.txt").write_text("public\n", encoding="utf-8")
    (repository / "AGENTS.md").write_text("private plan\n", encoding="utf-8")
    ignored_directory = repository / "reports"
    ignored_directory.mkdir()
    (ignored_directory / "internal.md").write_text("private\n", encoding="utf-8")
    _git(repository, "add", "kept.txt", "AGENTS.md")
    _git(repository, "add", "--force", "reports/internal.md")

    (repository / ".gitignore").write_text("reports/\n", encoding="utf-8")
    (repository / "new.txt").write_text("new public file\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    return repository


def test_collect_export_paths_reapplies_ignore_rules_to_tracked_files(
    source_repository: Path,
) -> None:
    export_paths, ignored_paths, withheld_paths = exporter.collect_export_paths(
        source_repository
    )

    assert {path.as_posix() for path in export_paths} == {
        ".gitignore",
        "kept.txt",
        "new.txt",
    }
    assert [path.as_posix() for path in ignored_paths] == ["reports/internal.md"]
    assert [path.as_posix() for path in withheld_paths] == ["AGENTS.md"]


def test_export_public_snapshot_copies_only_reviewable_files(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "public"

    result = exporter.export_public_snapshot(source_repository, destination)

    assert result.copied_files == 3
    assert result.ignored_tracked_files == 1
    assert result.withheld_files == 1
    assert not (destination / "AGENTS.md").exists()
    assert (destination / "kept.txt").read_text(encoding="utf-8") == "public\n"
    assert (destination / "new.txt").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / "reports").exists()


@pytest.mark.parametrize("staged", [False, True], ids=["unstaged", "staged"])
def test_export_public_snapshot_preserves_worktree_deletions(
    source_repository: Path,
    tmp_path: Path,
    staged: bool,
) -> None:
    """A working-tree snapshot omits deleted files before and after staging."""
    (source_repository / "kept.txt").unlink()
    if staged:
        _git(source_repository, "add", "--update", "kept.txt")
    before_index = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=source_repository
    )
    destination = tmp_path / "public"

    result = exporter.export_public_snapshot(source_repository, destination)

    assert result.copied_files == 2
    assert result.ignored_tracked_files == 1
    assert not (destination / "kept.txt").exists()
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new public file\n"
    assert (destination / ".gitignore").is_file()
    assert not (destination / ".git").exists()
    assert (
        subprocess.check_output(
            ["git", "ls-files", "--stage", "-z"], cwd=source_repository
        )
        == before_index
    )


def test_export_public_snapshot_refuses_existing_destination(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "public"
    destination.mkdir()

    with pytest.raises(exporter.SnapshotExportError, match="already exists"):
        exporter.export_public_snapshot(source_repository, destination)


def test_export_public_snapshot_refuses_destination_inside_source(
    source_repository: Path,
) -> None:
    with pytest.raises(exporter.SnapshotExportError, match="outside"):
        exporter.export_public_snapshot(source_repository, source_repository / "public")


def test_export_public_snapshot_refuses_symlinks(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    link = source_repository / "linked.txt"
    try:
        link.symlink_to(source_repository / "kept.txt")
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform")

    with pytest.raises(exporter.SnapshotExportError, match="Symbolic links"):
        exporter.export_public_snapshot(source_repository, tmp_path / "public")


def test_export_public_snapshot_refuses_unresolved_lfs_pointers(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    pointer = source_repository / "plot.png"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        "size 1234\n",
        encoding="utf-8",
    )
    _git(source_repository, "add", "plot.png")
    destination = tmp_path / "public"

    with pytest.raises(
        exporter.SnapshotExportError,
        match=r"Unresolved Git LFS pointer: plot\.png",
    ):
        exporter.export_public_snapshot(source_repository, destination)

    assert not destination.exists()


def test_tracked_dangling_symlink_is_rejected_as_a_link(
    source_repository: Path,
    tmp_path: Path,
) -> None:
    """A tracked link with an absent target must not become an omitted deletion."""
    kept = source_repository / "kept.txt"
    kept.unlink()
    try:
        kept.symlink_to(tmp_path / "absent.txt")
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform")

    destination = tmp_path / "public"
    with pytest.raises(exporter.SnapshotExportError, match="Symbolic links"):
        exporter.export_public_snapshot(source_repository, destination)

    assert not destination.exists()


def test_candidate_disappearing_after_git_selection_remains_an_error(
    source_repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only Git-reported deletions may be omitted from the snapshot."""
    original_ignore_check = exporter._is_ignored

    def remove_after_selection(root, relative_path):
        if relative_path == "kept.txt":
            (root / relative_path).unlink()
        return original_ignore_check(root, relative_path)

    monkeypatch.setattr(exporter, "_is_ignored", remove_after_selection)
    destination = tmp_path / "public"
    with pytest.raises(exporter.SnapshotExportError, match="missing.*kept.txt"):
        exporter.export_public_snapshot(source_repository, destination)

    assert not destination.exists()
