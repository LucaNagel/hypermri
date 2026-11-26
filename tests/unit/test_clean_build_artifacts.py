"""Cleanup boundary regressions using isolated, disposable Git worktrees."""

import errno
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "clean_build_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("hypermri_artifact_cleanup", _SCRIPT)
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


_SYMLINK_UNAVAILABLE_ERRNOS = {
    errno.EACCES,
    errno.EPERM,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
}


def _create_symlink(path, target, *, target_is_directory=False):
    try:
        path.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if (
            exc.errno in _SYMLINK_UNAVAILABLE_ERRNOS
            or getattr(exc, "winerror", None) == 1314
        ):
            pytest.skip(f"Symbolic links are unavailable on this platform: {exc}")
        raise


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def _create_file(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"invented generated artifact\n")
    return path


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / ".gitignore").write_text(
        "build/\ndist/\n*.egg-info/\n.tox/\n.nox/\n.pytest_cache/\n"
        "htmlcov/\n.coverage\ncoverage.xml\ndocs/_build/\nresearch/\nplans/\n",
        encoding="utf-8",
    )
    _git(root, "add", ".gitignore")
    return root


def test_packaging_selection_retains_its_existing_scope(repository):
    for name in (
        "build/item",
        "dist/item",
        "pkg.egg-info/item",
        "src/pkg.egg-info/item",
        ".tox/item",
    ):
        _create_file(repository, name)
    paths = cleanup.build_artifact_paths(repository)
    assert [path.relative_to(repository).as_posix() for path in paths] == [
        "build",
        "dist",
        "pkg.egg-info",
        "src/pkg.egg-info",
    ]


def test_default_cleanup_is_a_read_only_preview(repository):
    artifact = _create_file(repository, "build/generated")
    before = _git(repository, "status", "--porcelain", "-z")
    assert cleanup.clean_build_artifacts(repository) == [repository / "build"]
    assert artifact.read_bytes() == b"invented generated artifact\n"
    assert _git(repository, "status", "--porcelain", "-z") == before


def test_explicit_dry_run_preserves_existing_artifacts(repository):
    artifact = _create_file(repository, "build/generated")
    assert cleanup.clean_build_artifacts(repository, dry_run=True) == [
        repository / "build"
    ]
    assert artifact.exists()


def test_explicit_packaging_cleanup_keeps_developer_and_other_ignored_content(
    repository,
):
    for name in (
        "build/item",
        "dist/item",
        "pkg.egg-info/item",
        "src/pkg.egg-info/item",
    ):
        _create_file(repository, name)
    preserved = [
        _create_file(repository, name)
        for name in (
            ".tox/item",
            "docs/_build/index.html",
            "research/raw",
            "plans/owner.txt",
            "src/code.py",
        )
    ]
    before = _git(repository, "status", "--porcelain", "-z")
    removed = cleanup.clean_build_artifacts(repository, dry_run=False)
    assert len(removed) == 4
    assert all(not path.exists() for path in removed)
    assert all(
        path.read_bytes() == b"invented generated artifact\n" for path in preserved
    )
    assert _git(repository, "status", "--porcelain", "-z") == before


def test_developer_cleanup_includes_only_known_artifacts(repository):
    names = (
        ".tox/item",
        ".nox/item",
        ".pytest_cache/item",
        "htmlcov/index.html",
        ".coverage",
        "coverage.xml",
        "docs/_build/index.html",
    )
    for name in names:
        _create_file(repository, name)
    preserved = [
        _create_file(repository, name)
        for name in (
            "research/raw",
            "plans/owner.txt",
            "src/code.py",
            "docs/source/page.rst",
        )
    ]
    expected = [
        repository / name
        for name in (
            ".coverage",
            ".nox",
            ".pytest_cache",
            ".tox",
            "coverage.xml",
            "docs/_build",
            "htmlcov",
        )
    ]
    assert cleanup.clean_build_artifacts(repository, include_developer=True) == sorted(
        expected
    )
    assert all(path.exists() for path in expected)
    assert cleanup.clean_build_artifacts(
        repository, include_developer=True, dry_run=False
    ) == sorted(expected)
    assert all(not path.exists() for path in expected)
    assert all(path.exists() for path in preserved)
    assert (
        cleanup.clean_build_artifacts(repository, include_developer=True, dry_run=False)
        == []
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_tracked_content_blocks_whole_selection_before_any_removal(repository, dry_run):
    safe = _create_file(repository, "build/generated")
    tracked = _create_file(repository, "dist/keep")
    _git(repository, "add", "--force", "dist/keep")
    before = _git(repository, "status", "--porcelain", "-z")
    with pytest.raises(ValueError, match="tracked or unignored.*dist"):
        cleanup.clean_build_artifacts(repository, dry_run=dry_run)
    assert safe.exists() and tracked.exists()
    assert _git(repository, "status", "--porcelain", "-z") == before


def test_non_ignored_artifact_is_never_removed(repository):
    (repository / ".gitignore").write_text("", encoding="utf-8")
    preserved = _create_file(repository, "build/source.py")
    with pytest.raises(ValueError, match="non-ignored artifact: build"):
        cleanup.clean_build_artifacts(repository, dry_run=False)
    assert preserved.exists()


@pytest.mark.parametrize("existing_target", [False, True])
def test_artifact_root_symlink_is_rejected_even_when_dangling(
    repository, tmp_path, existing_target
):
    target = tmp_path / "external"
    if existing_target:
        _create_file(target, "keep")
    _create_symlink(repository / "build", target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link: build"):
        cleanup.clean_build_artifacts(repository, dry_run=False)
    assert (repository / "build").is_symlink()
    if existing_target:
        assert (target / "keep").exists()


@pytest.mark.parametrize("ancestor", ["src", "docs"])
def test_ancestor_symlinks_are_rejected(repository, tmp_path, ancestor):
    target = tmp_path / "external"
    protected = _create_file(
        target, "pkg.egg-info/keep" if ancestor == "src" else "_build/keep"
    )
    _create_symlink(repository / ancestor, target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        cleanup.clean_build_artifacts(repository, include_developer=True, dry_run=False)
    assert protected.exists()


def test_nested_symbolic_links_are_unlinked_without_following_targets(
    repository, tmp_path
):
    generated = _create_file(repository, "build/generated")
    preserved = _create_file(tmp_path / "external", "keep")
    _create_symlink(
        repository / "build" / "link", preserved.parent, target_is_directory=True
    )
    _create_symlink(repository / "build" / "broken", tmp_path / "missing")
    cleanup.clean_build_artifacts(repository, dry_run=False)
    assert not generated.exists()
    assert preserved.read_bytes() == b"invented generated artifact\n"


@pytest.mark.parametrize("git_marker", ["file", "directory"])
def test_nested_git_checkouts_are_preserved(repository, git_marker):
    preserved = _create_file(repository, "build/checkout/source.py")
    marker = preserved.parent / ".git"
    if git_marker == "file":
        marker.write_text("gitdir: /invented/worktree\n", encoding="utf-8")
    else:
        marker.mkdir()
    with pytest.raises(ValueError, match="nested Git repository"):
        cleanup.clean_build_artifacts(repository, dry_run=False)
    assert preserved.exists() and marker.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_mounted_directories_are_preserved(repository, monkeypatch, nested):
    preserved = _create_file(repository, "build/mounted/keep")
    mounted = preserved.parent if nested else repository / "build"
    original = Path.is_mount
    monkeypatch.setattr(
        Path, "is_mount", lambda path: path == mounted or original(path)
    )
    with pytest.raises(ValueError, match="mounted"):
        cleanup.clean_build_artifacts(repository, dry_run=False)
    assert preserved.exists()


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable on this platform"
)
def test_non_regular_artifact_root_is_preserved(repository):
    path = repository / ".coverage"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="non-regular artifact"):
        cleanup.clean_build_artifacts(repository, include_developer=True, dry_run=False)
    assert path.exists()


@pytest.mark.parametrize("inside_repository", [False, True])
def test_cleanup_requires_the_worktree_root(repository, tmp_path, inside_repository):
    root = repository / "src" if inside_repository else tmp_path / "not_git"
    root.mkdir()
    preserved = _create_file(root, "build/keep")
    with pytest.raises(ValueError, match="worktree root"):
        cleanup.clean_build_artifacts(root, dry_run=False)
    assert preserved.exists()


@pytest.mark.parametrize("operation", ["check-ignore", "ls-files"])
def test_failed_git_checks_block_removal(repository, monkeypatch, operation):
    preserved = _create_file(repository, "build/keep")
    original = cleanup._run_git

    def failed_git(root, arguments):
        if arguments[0] == operation:
            return subprocess.CompletedProcess(
                arguments, 128, b"", b"invented Git failure"
            )
        return original(root, arguments)

    monkeypatch.setattr(cleanup, "_run_git", failed_git)
    with pytest.raises(RuntimeError, match="Cannot check.*invented Git failure"):
        cleanup.clean_build_artifacts(repository, dry_run=False)
    assert preserved.exists()


def test_egg_info_names_use_literal_git_paths(repository):
    generated = _create_file(repository, "pkg[1].egg-info/generated")
    tracked = _create_file(repository, "pkg1.egg-info/keep")
    _git(repository, "add", "--force", "pkg1.egg-info/keep")
    # Select the bracketed name directly to check its tracking query in isolation.
    cleanup._validate_artifact(repository, generated.parent)
    assert tracked.exists()


@pytest.mark.parametrize("args", [[], ["--dry-run"], ["--all"]])
def test_cli_previews_by_default(repository, monkeypatch, capsys, args):
    generated = _create_file(repository, "build/generated")
    developer = _create_file(repository, ".tox/generated")
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", repository)
    assert cleanup.main(args) == 0
    output = capsys.readouterr().out
    assert "Would remove build" in output
    assert ("Would remove .tox" in output) == ("--all" in args)
    assert generated.exists() and developer.exists()


def test_cli_apply_is_explicit_and_scoped(repository, monkeypatch, capsys):
    generated = _create_file(repository, "build/generated")
    developer = _create_file(repository, ".tox/generated")
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", repository)
    assert cleanup.main(["--apply"]) == 0
    assert "Removed build" in capsys.readouterr().out
    assert not generated.exists() and developer.exists()


def test_cli_safety_failure_is_actionable_and_preserves_files(
    repository, monkeypatch, capsys
):
    generated = _create_file(repository, "build/generated")
    _git(repository, "add", "--force", "build/generated")
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", repository)
    assert cleanup.main(["--apply"]) == 1
    assert "Artifact cleanup failed:" in capsys.readouterr().err
    assert generated.exists()


def test_cli_empty_selection_is_successful(repository, monkeypatch, capsys):
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", repository)
    assert cleanup.main([]) == 0
    assert "No selected artifacts found." in capsys.readouterr().out


def test_cli_rejects_conflicting_modes():
    with pytest.raises(SystemExit) as error:
        cleanup.main(["--apply", "--dry-run"])
    assert error.value.code == 2
