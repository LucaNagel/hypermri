#!/usr/bin/env python3
"""Preview or explicitly remove ignored, untracked local developer artifacts.

The default command previews packaging output. ``--all`` also selects known
test, coverage and documentation artifacts, and ``--apply`` permits deletion.
Git worktree, ignore and tracked-file checks guard the complete selected set
before any removal. Artifact roots/ancestors cannot be symbolic links; nested
links are unlinked without following their targets. Nested repositories and
mounted directories are rejected. No MRI data or scientific operation is read.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEVELOPER_ARTIFACTS = (
    ".tox",
    ".nox",
    ".pytest_cache",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "docs/_build",
)


def build_artifact_paths(
    project_root: Path, *, include_developer: bool = False
) -> list[Path]:
    """List known packaging and optionally developer artifact candidates.

    Parameters
    ----------
    project_root : pathlib.Path
        Project directory. Paths are normalized to its absolute location.
    include_developer : bool, default False
        Also include .tox, .nox, .pytest_cache, htmlcov, .coverage, coverage.xml
        and docs/_build. Packaging candidates are build, dist and egg-info
        directories immediately under the root or src.

    Returns
    -------
    list of pathlib.Path
        Sorted absolute candidates. Fixed paths may be missing; egg-info names
        are discovered on the filesystem. No deletion or Git mutation occurs.

    Raises
    ------
    ValueError
        If src is a symbolic link, so egg-info discovery would leave the
        project directory tree.

    Notes
    -----
    Reads directory names only, deterministically, without scientific data or
    caches. :func:`clean_build_artifacts` validates candidates against Git and
    filesystem boundaries before acting on them.
    """
    project_root = project_root.resolve()
    if (project_root / "src").is_symlink():
        raise ValueError("Cannot discover egg-info through a symbolic link: src")
    paths = [
        project_root / "build",
        project_root / "dist",
    ]
    paths.extend(project_root.glob("*.egg-info"))
    paths.extend((project_root / "src").glob("*.egg-info"))
    if include_developer:
        paths.extend(project_root / name for name in _DEVELOPER_ARTIFACTS)
    return sorted(paths)


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, check=False
    )


def _validate_repository(project_root: Path) -> Path:
    root = project_root.resolve()
    result = _run_git(root, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise ValueError(f"Cleanup requires a Git worktree root: {root}")
    discovered_root = Path(os.fsdecode(result.stdout).strip()).resolve()
    if discovered_root != root:
        raise ValueError(
            f"Cleanup requires the worktree root {discovered_root}, got {root}"
        )
    return root


def _validate_artifact(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"Refusing cleanup through symbolic link: {relative}")
    if path.is_mount():
        raise ValueError(f"Refusing cleanup of mounted artifact: {relative}")
    ignored = _run_git(
        root, ["check-ignore", "--no-index", "--quiet", "--", relative.as_posix()]
    )
    if ignored.returncode == 1:
        raise ValueError(f"Refusing cleanup of non-ignored artifact: {relative}")
    if ignored.returncode != 0:
        raise RuntimeError(
            f"Cannot check ignore rules for {relative}: {os.fsdecode(ignored.stderr).strip()}"
        )
    protected = _run_git(
        root,
        [
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            f":(literal){relative.as_posix()}",
        ],
    )
    if protected.returncode != 0:
        raise RuntimeError(
            f"Cannot check tracked/unignored files in {relative}: {os.fsdecode(protected.stderr).strip()}"
        )
    if protected.stdout:
        raise ValueError(
            f"Refusing cleanup of artifact with tracked or unignored files: {relative}"
        )
    if not path.is_dir():
        if not path.is_file():
            raise ValueError(f"Refusing cleanup of non-regular artifact: {relative}")
        return

    def fail_walk(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(
        path, followlinks=False, onerror=fail_walk
    ):
        if ".git" in directories or ".git" in files:
            raise ValueError(f"Refusing cleanup of nested Git repository in {relative}")
        if any((Path(directory) / name).is_mount() for name in directories):
            raise ValueError(f"Refusing cleanup of mounted directory in {relative}")


def clean_build_artifacts(
    project_root: Path, *, dry_run: bool = True, include_developer: bool = False
) -> list[Path]:
    """Preview or remove validated local packaging/developer artifacts.

    Parameters
    ----------
    project_root : pathlib.Path
        Actual Git worktree root. Only the known artifact paths beneath it
        are eligible; arbitrary ignored directories are never selected.
    dry_run : bool, default True
        Return eligible paths without deleting them. False explicitly enables
        removal of the selected existing files/directories.
    include_developer : bool, default False
        Include the developer paths documented by :func:`build_artifact_paths`.

    Returns
    -------
    list of pathlib.Path
        Sorted absolute paths that existed and passed validation. With dry_run
        False these paths have been removed, including their generated contents.

    Raises
    ------
    ValueError
        If root is not the worktree root, a candidate is not ignored or contains
        tracked/unignored files, a root/ancestor is a symbolic link, or a
        candidate contains a nested repository, mount or non-regular root.
    RuntimeError
        If Git cannot check candidate ignore/tracking state.
    OSError
        If directory inspection or removal fails. Removal failures may leave
        a partially cleaned set; validation failures remove nothing.

    Notes
    -----
    Validates the whole existing set before deletion and revalidates each
    candidate immediately before removal. Does not mutate Git, read MRI values
    or follow nested symbolic links. Concurrent filesystem changes are not an
    atomic transaction. Results are deterministic for a fixed filesystem/Git
    state; no cache is introduced.

    Examples
    --------
    Preview artifacts, then explicitly clear that selection::

        paths = clean_build_artifacts(root, include_developer=True)
        removed = clean_build_artifacts(
            root, dry_run=False, include_developer=True,
        )
    """
    root = _validate_repository(project_root)
    candidates = build_artifact_paths(root, include_developer=include_developer)
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    for path in existing:
        _validate_artifact(root, path)
    if dry_run:
        return existing
    for path in existing:
        _validate_artifact(root, path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return existing


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact preview/removal command for this checkout.

    Parameters
    ----------
    argv : sequence of str or None, optional
        CLI arguments. None reads the process command line. --all selects
        developer artifacts, --apply enables deletion, and --dry-run explicitly
        selects the default preview; --apply and --dry-run are mutually exclusive.

    Returns
    -------
    int
        Zero after a successful preview/removal, one after a safety or I/O error.
        Operational errors are written to stderr without deleting on failed
        preflight; the selected relative paths are printed to stdout.

    Raises
    ------
    SystemExit
        For argparse help or invalid CLI arguments.

    Notes
    -----
    No Git or scientific-data mutation. Removal is possible only with --apply.
    Uses the fixed checkout root containing this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="Remove validated artifacts explicitly."
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print artifact paths without deleting them.",
    )
    parser.add_argument(
        "--all",
        dest="include_developer",
        action="store_true",
        help="Also select test environments, coverage and documentation builds.",
    )
    args = parser.parse_args(argv)

    try:
        removed = clean_build_artifacts(
            PROJECT_ROOT,
            dry_run=not args.apply,
            include_developer=args.include_developer,
        )
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Artifact cleanup failed: {error}", file=sys.stderr)
        return 1
    action = "Removed" if args.apply else "Would remove"
    if not removed:
        print("No selected artifacts found.")
        return 0
    for path in removed:
        print(f"{action} {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
