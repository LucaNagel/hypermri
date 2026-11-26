#!/usr/bin/env python3
"""Export a clean public snapshot without copying private Git history."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
# Tracked files that serve private development only. The agent instructions
# point to the private plan and tracker, which are never published.
PRIVATE_DEVELOPMENT_FILES = frozenset({PurePosixPath("AGENTS.md")})


class SnapshotExportError(RuntimeError):
    """Raised when a safe public-snapshot export cannot be completed."""


@dataclass(frozen=True)
class ExportResult:
    """Summary of one completed snapshot export."""

    destination: Path
    copied_files: int
    ignored_tracked_files: int
    withheld_files: int


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """Run a read-only Git command in ``repository_root``."""
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=text,
    )


def _validate_repository(repository_root: Path) -> Path:
    """Return a resolved Git worktree root or raise a clear error."""
    root = repository_root.resolve()
    result = _run_git(root, ["rev-parse", "--show-toplevel"], text=True)
    if result.returncode != 0:
        raise SnapshotExportError(f"Not a Git worktree: {root}")
    discovered_root = Path(result.stdout.strip()).resolve()
    if discovered_root != root:
        raise SnapshotExportError(
            f"Source must be the worktree root ({discovered_root}), got {root}"
        )
    return root


def _is_ignored(repository_root: Path, relative_path: str) -> bool:
    """Apply current ignore rules even when ``relative_path`` is tracked."""
    result = _run_git(
        repository_root,
        ["check-ignore", "--no-index", "--quiet", "--", relative_path],
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    error = os.fsdecode(result.stderr).strip()
    raise SnapshotExportError(
        f"Could not evaluate ignore rules for {relative_path!r}: {error}"
    )


def _is_git_lfs_pointer(path: Path) -> bool:
    """Return whether path is an unresolved Git LFS pointer."""
    try:
        with path.open("rb") as stream:
            header = stream.read(len(GIT_LFS_POINTER_HEADER))
    except OSError as exc:
        raise SnapshotExportError(
            f"Could not inspect snapshot candidate {path.name!r}: {exc}"
        ) from exc
    return header == GIT_LFS_POINTER_HEADER


def collect_export_paths(
    repository_root: Path,
) -> tuple[list[PurePosixPath], list[PurePosixPath], list[PurePosixPath]]:
    """Select existing working-tree files for a history-free snapshot.

    Parameters
    ----------
    repository_root : pathlib.Path
        Git worktree root whose current files are selected. Relative paths are
        resolved against the current working directory.

    Returns
    -------
    tuple of list of pathlib.PurePosixPath
        Sorted repository-relative export paths, tracked paths excluded by
        current ignore rules, and withheld private development files listed in
        ``PRIVATE_DEVELOPMENT_FILES``. Tracked files reported as deleted by Git
        are omitted from all lists; staged deletions are already absent from the
        candidate list. New non-ignored files and current tracked-file contents
        are included regardless of staging.

    Raises
    ------
    SnapshotExportError
        If the source is not a worktree root, Git cannot enumerate candidates or
        deletions, or a remaining candidate is unsafe, a symbolic link, an
        unresolved Git LFS pointer, unreadable or no longer a regular file.
    OSError
        If the Git executable cannot be started.

    Notes
    -----
    Reads Git metadata and file headers eagerly without changing the index or
    worktree. This is a working-tree export, not an atomic filesystem snapshot;
    keep source files unchanged during collection and copying.
    """
    root = _validate_repository(repository_root)
    result = _run_git(
        root,
        ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    )
    if result.returncode != 0:
        error = os.fsdecode(result.stderr).strip()
        raise SnapshotExportError(f"Could not list snapshot candidates: {error}")

    deleted = _run_git(root, ["ls-files", "--deleted", "-z"])
    if deleted.returncode != 0:
        error = os.fsdecode(deleted.stderr).strip()
        raise SnapshotExportError(f"Could not list working-tree deletions: {error}")
    deleted_paths = set(deleted.stdout.split(b"\0"))

    export_paths: list[PurePosixPath] = []
    ignored_paths: list[PurePosixPath] = []
    withheld_paths: list[PurePosixPath] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path or raw_path in deleted_paths:
            continue
        path_text = os.fsdecode(raw_path)
        relative_path = PurePosixPath(path_text)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or ".git" in relative_path.parts
        ):
            raise SnapshotExportError(f"Unsafe repository path: {path_text!r}")
        if _is_ignored(root, path_text):
            ignored_paths.append(relative_path)
            continue
        if relative_path in PRIVATE_DEVELOPMENT_FILES:
            withheld_paths.append(relative_path)
            continue

        source_path = root.joinpath(*relative_path.parts)
        if source_path.is_symlink():
            raise SnapshotExportError(
                f"Symbolic links require explicit review: {path_text}"
            )
        if not source_path.is_file():
            raise SnapshotExportError(
                f"Snapshot candidate is missing or not a regular file: {path_text}"
            )
        if _is_git_lfs_pointer(source_path):
            raise SnapshotExportError(
                f"Unresolved Git LFS pointer: {path_text}. "
                "Run git lfs pull with an explicit empty --exclude filter before exporting."
            )
        export_paths.append(relative_path)

    return sorted(export_paths), sorted(ignored_paths), sorted(withheld_paths)


def export_public_snapshot(
    repository_root: Path,
    destination: Path,
) -> ExportResult:
    """Copy the current working tree into a new, history-free directory.

    Parameters
    ----------
    repository_root : pathlib.Path
        Git worktree root containing the prospective snapshot. Current tracked
        files and non-ignored new files are copied; deleted files and private
        development files are omitted.
    destination : pathlib.Path
        New directory outside the source worktree with an existing parent.
        User-home expansion and relative-path resolution are applied.

    Returns
    -------
    ExportResult
        Resolved destination, copied-file count, count of existing tracked
        files excluded by current ignore rules and count of withheld private
        development files.

    Raises
    ------
    SnapshotExportError
        If the source or destination violates the export contract or selection
        finds unsafe, unavailable or unresolved candidates.
    OSError
        If Git cannot be started, directory creation fails or copying fails.

    Notes
    -----
    Copies files and their metadata eagerly without changing Git state or
    approving publication. No Git history or ignored content is copied. A
    copying failure removes the newly created destination. Keep the source
    unchanged during this non-atomic operation.
    """
    root = _validate_repository(repository_root)
    target = destination.expanduser().resolve()
    if target.exists():
        raise SnapshotExportError(f"Destination already exists: {target}")
    if root == target or root in target.parents:
        raise SnapshotExportError(
            "Destination must be outside the private repository worktree"
        )
    if not target.parent.is_dir():
        raise SnapshotExportError(f"Destination parent does not exist: {target.parent}")

    export_paths, ignored_paths, withheld_paths = collect_export_paths(root)
    target.mkdir()
    try:
        for relative_path in export_paths:
            source_path = root.joinpath(*relative_path.parts)
            target_path = target.joinpath(*relative_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
    except Exception:
        shutil.rmtree(target)
        raise

    return ExportResult(
        destination=target,
        copied_files=len(export_paths),
        ignored_tracked_files=len(ignored_paths),
        withheld_files=len(withheld_paths),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy the reviewed working tree to a new directory without .git "
            "history, ignored files, or symlinks."
        )
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="new destination directory; it must not already exist",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT,
        help="private Git worktree root (defaults to this repository)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export the snapshot selected by command-line arguments."""
    args = build_parser().parse_args(argv)
    try:
        result = export_public_snapshot(args.source, args.destination)
    except SnapshotExportError as exc:
        print(f"Public snapshot export failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {result.copied_files} file(s) to {result.destination}. "
        f"Excluded {result.ignored_tracked_files} tracked file(s) because they "
        "match current ignore rules. "
        f"Withheld {result.withheld_files} private development file(s)."
    )
    print("The destination has no Git history. Initialize and audit it separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
