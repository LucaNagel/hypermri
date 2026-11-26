#!/usr/bin/env python3
"""Audit the prospective public repository snapshot for sensitive content."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_EMAILS = frozenset(
    {
        "git@github.com",
        "luca.nagel@posteo.de",
        "luca.nagel@protonmail.com",
        "noreply@hypermri.invalid",
    }
)

DOCUMENTATION_MEDIA_ROOT = PurePosixPath("docs/source/_static/media")
ALLOWED_DOCUMENTATION_MEDIA_SUFFIXES = frozenset({".mov", ".mp4", ".png"})
ALLOWED_DOCUMENTATION_MEDIA_FILES = frozenset({DOCUMENTATION_MEDIA_ROOT / "README.md"})
ALLOWED_DATA_FILES: frozenset[PurePosixPath] = frozenset(
    {
        PurePosixPath("tests/fixtures/synthetic_csi.py"),
        PurePosixPath("tests/fixtures/synthetic_metabolite_detection.py"),
        PurePosixPath("tests/fixtures/synthetic_noise_quantification.py"),
        PurePosixPath("tests/fixtures/synthetic_spectroscopy_uncertainty.py"),
        PurePosixPath("tests/fixtures/synthetic_t1.py"),
    }
)

RAW_SCANNER_NAMES = frozenset(
    {
        "2dseq",
        "acqu",
        "acqu.par",
        "acqus",
        "acqu2",
        "acqu2s",
        "acqu3",
        "acqu3s",
        "acqp",
        "data.1d",
        "data.2d",
        "data.3d",
        "dicomdir",
        "fid",
        "method",
        "proc.par",
        "proc2s",
        "proc3s",
        "proc4s",
        "procpar",
        "procs",
        "rawdata.job0",
        "reco",
        "ser",
        "subject",
        "visu_pars",
    }
)
RAW_OR_DATA_SUFFIXES = (
    ".7z",
    ".bin",
    ".dat",
    ".dcm",
    ".h5",
    ".hdf5",
    ".ima",
    ".joblib",
    ".mat",
    ".mha",
    ".mhd",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".nrrd",
    ".parquet",
    ".pickle",
    ".pkl",
    ".raw",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".twix",
    ".zip",
)
PRIVATE_KEY_SUFFIXES = frozenset({".jks", ".key", ".keystore", ".p12", ".pfx", ".ppk"})
PRIVATE_KEY_NAMES = frozenset({"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"})
PRIVATE_CONFIG_NAMES = frozenset(
    {
        ".envrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "service-account.json",
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
    }
)
SAFE_ENV_TEMPLATE_MARKERS = frozenset({"example", "sample", "template"})
TEST_DATA_DIRECTORIES = frozenset(
    {"fixture_data", "fixtures_data", "test_data", "testdata"}
)
SAFE_TEST_DATA_DOCUMENT_NAMES = frozenset(
    {"license", "license.md", "readme", "readme.md", "readme.rst", "readme.txt"}
)
PLACEHOLDER_HOME_NAMES = frozenset(
    {"ci", "docs", "runner", "runneradmin", "test", "user", "username"}
)

EMAIL_RE = re.compile(
    r"(?<![A-Z0-9._%+\-])"
    r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})",
    re.IGNORECASE,
)
OLD_REPOSITORY_RE = re.compile(
    r"(?:https?://|ssh://git@|git@)?"
    r"github\.com[:/]schillinglab/hypermri(?:\.git)?",
    re.IGNORECASE,
)
UNIX_HOME_RE = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/(?P<name>[A-Za-z0-9._-]+)")
WINDOWS_HOME_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/mnt/[A-Za-z]/)"
    r"Users[\\/](?P<name>[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN "
    r"(?:RSA |OPENSSH |EC |DSA |PGP |ENCRYPTED )?"
    r"PRIVATE\s+KEY(?: BLOCK)?-----"
)


@dataclass(frozen=True)
class Finding:
    """One public-snapshot policy violation."""

    path: str
    code: str
    message: str
    line: int | None = None

    def format(self) -> str:
        """Return a concise message without echoing the sensitive value."""
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{location}: [{self.code}] {self.message}"


def normalize_relative_path(path: str | PurePosixPath) -> PurePosixPath:
    """Return a normalized repository-relative POSIX path."""
    normalized = PurePosixPath(str(path).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Expected a repository-relative path, got {path!s}")
    return normalized


def is_documentation_media_path(path: str | PurePosixPath) -> bool:
    """Return whether path is inside the documentation media tree."""
    relative_path = normalize_relative_path(path)
    return relative_path.is_relative_to(DOCUMENTATION_MEDIA_ROOT)


def is_allowed_data_path(path: str | PurePosixPath) -> bool:
    """Return whether raw-like data is explicitly allowed at ``path``."""
    relative_path = normalize_relative_path(path)
    if relative_path in ALLOWED_DATA_FILES:
        return True
    return relative_path in ALLOWED_DOCUMENTATION_MEDIA_FILES or (
        is_documentation_media_path(relative_path)
        and relative_path.suffix.casefold() in ALLOWED_DOCUMENTATION_MEDIA_SUFFIXES
    )


def is_environment_file(path: str | PurePosixPath) -> bool:
    """Return whether ``path`` looks like a private environment file."""
    name = normalize_relative_path(path).name.casefold()
    parts = set(name.replace("-", ".").replace("_", ".").split("."))
    if parts & SAFE_ENV_TEMPLATE_MARKERS:
        return False
    return (
        name == ".env"
        or name.startswith(".env.")
        or name.endswith(".env")
        or name in PRIVATE_CONFIG_NAMES
    )


def _has_test_data_directory(path: PurePosixPath) -> bool:
    """Return whether a path is inside a conventional test-data directory."""
    parts = tuple(part.casefold() for part in path.parts)
    if any(part in TEST_DATA_DIRECTORIES for part in parts[:-1]):
        return True
    return any(
        left in {"test", "tests"} and right in {"data", "fixtures"}
        for left, right in zip(parts, parts[1:], strict=False)
    )


def find_path_violations(path: str | PurePosixPath) -> list[Finding]:
    """Return violations detectable from a repository-relative path alone."""
    relative_path = normalize_relative_path(path)
    path_text = relative_path.as_posix()
    name = relative_path.name.casefold()
    lower_path = path_text.casefold()
    findings: list[Finding] = []

    if is_environment_file(relative_path):
        findings.append(
            Finding(
                path_text,
                "private-environment-file",
                "private environment or credential file must not be published",
            )
        )

    if (
        name in PRIVATE_KEY_NAMES
        or relative_path.suffix.casefold() in PRIVATE_KEY_SUFFIXES
    ):
        findings.append(
            Finding(
                path_text,
                "private-key-file",
                "private-key container or key filename must not be published",
            )
        )

    if path_text == "AGENTS.md":
        findings.append(
            Finding(
                path_text,
                "private-development-file",
                "agent instructions refer to the private plan and must not be "
                "published; export with scripts/export_public_snapshot.py",
            )
        )

    allowed_data_path = is_allowed_data_path(relative_path)
    if is_documentation_media_path(relative_path) and not allowed_data_path:
        findings.append(
            Finding(
                path_text,
                "unapproved-documentation-media",
                "file type is not an explicitly approved documentation web-media format",
            )
        )

    if allowed_data_path:
        return findings

    has_raw_name = name in RAW_SCANNER_NAMES
    has_raw_directory = any(
        part.casefold().endswith((".fid", ".zarr")) for part in relative_path.parts
    )
    has_data_suffix = lower_path.endswith(RAW_OR_DATA_SUFFIXES)
    if has_raw_name or has_raw_directory or has_data_suffix:
        findings.append(
            Finding(
                path_text,
                "raw-or-data-file",
                "raw scanner, archive, or binary data is outside an allowed path",
            )
        )

    if relative_path.suffix.casefold() == ".ipynb":
        findings.append(
            Finding(
                path_text,
                "unapproved-notebook",
                "notebook is outside the explicitly approved tutorial path",
            )
        )

    if (
        _has_test_data_directory(relative_path)
        and name not in SAFE_TEST_DATA_DOCUMENT_NAMES
    ):
        findings.append(
            Finding(
                path_text,
                "test-data-file",
                "test fixture or test data is outside an allowed public path",
            )
        )

    return findings


def _contains_personal_home(text: str) -> bool:
    """Return whether text contains a non-placeholder user home path."""
    for pattern in (UNIX_HOME_RE, WINDOWS_HOME_RE):
        for match in pattern.finditer(text):
            if match.group("name").casefold() not in PLACEHOLDER_HOME_NAMES:
                return True
    return False


def find_text_violations(
    path: str | PurePosixPath,
    text: str,
    *,
    start_line: int = 1,
) -> list[Finding]:
    """Return sensitive-content findings for decoded text."""
    relative_path = normalize_relative_path(path)
    path_text = relative_path.as_posix()
    findings: list[Finding] = []

    for line_number, line in enumerate(text.splitlines(), start=start_line):
        if OLD_REPOSITORY_RE.search(line):
            findings.append(
                Finding(
                    path_text,
                    "old-repository-link",
                    "link still points to the private schillinglab repository",
                    line_number,
                )
            )

        unapproved_email = any(
            match.group(1).casefold() not in ALLOWED_EMAILS
            for match in EMAIL_RE.finditer(line)
        )
        if unapproved_email:
            findings.append(
                Finding(
                    path_text,
                    "unapproved-email",
                    "email address is not approved for the public snapshot",
                    line_number,
                )
            )

        if _contains_personal_home(line):
            findings.append(
                Finding(
                    path_text,
                    "personal-absolute-path",
                    "personal absolute home path must not be published",
                    line_number,
                )
            )

        if PRIVATE_KEY_RE.search(line):
            findings.append(
                Finding(
                    path_text,
                    "private-key-content",
                    "private-key material must not be published",
                    line_number,
                )
            )

    return findings


def is_probably_binary(sample: bytes) -> bool:
    """Return whether a byte sample should not be decoded as repository text."""
    if not sample:
        return False
    if b"\0" in sample:
        return True
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    control_characters = sum(
        character < " " and character not in "\n\r\t\f\b" for character in decoded
    )
    return control_characters > max(1, len(decoded) // 100)


def audit_file(repository_root: Path, path: str | PurePosixPath) -> list[Finding]:
    """Audit one file without following symlinks or modifying repository state."""
    relative_path = normalize_relative_path(path)
    path_text = relative_path.as_posix()
    absolute_path = repository_root / Path(*relative_path.parts)
    findings = find_path_violations(relative_path)

    if not absolute_path.exists() and not absolute_path.is_symlink():
        return findings

    if absolute_path.is_symlink():
        target = os.readlink(absolute_path)
        if Path(target).is_absolute():
            findings.append(
                Finding(
                    path_text,
                    "absolute-symlink",
                    "absolute symlink target must not be published",
                )
            )
        findings.extend(find_text_violations(relative_path, target))
        return findings

    if not absolute_path.is_file():
        return findings

    try:
        with absolute_path.open("rb") as handle:
            sample = handle.read(8192)
        if is_probably_binary(sample):
            return findings

        with absolute_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                findings.extend(
                    find_text_violations(
                        relative_path,
                        line,
                        start_line=line_number,
                    )
                )
    except OSError:
        findings.append(
            Finding(
                path_text,
                "unreadable-file",
                "file could not be read during the public snapshot audit",
            )
        )

    return findings


def parse_git_path_output(output: bytes) -> list[PurePosixPath]:
    """Parse NUL-delimited paths returned by ``git ls-files -z``."""
    paths = {
        normalize_relative_path(os.fsdecode(item))
        for item in output.split(b"\0")
        if item
    }
    return sorted(paths, key=lambda path: path.as_posix())


def list_snapshot_paths(repository_root: Path) -> list[PurePosixPath]:
    """List tracked files plus unignored untracked files using Git."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return parse_git_path_output(result.stdout)


def audit_snapshot(
    repository_root: Path,
    paths: Iterable[str | PurePosixPath],
) -> list[Finding]:
    """Audit a supplied repository snapshot path list."""
    findings: list[Finding] = []
    for path in paths:
        findings.extend(audit_file(repository_root, path))
    return sorted(
        findings,
        key=lambda finding: (
            finding.path,
            finding.line or 0,
            finding.code,
            finding.message,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the read-only public snapshot audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Git worktree to audit (defaults to the project root).",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()

    try:
        paths = list_snapshot_paths(repository_root)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Could not enumerate the Git snapshot: {error}", file=sys.stderr)
        return 2

    findings = audit_snapshot(repository_root, paths)
    if findings:
        print(
            f"Public snapshot audit failed with {len(findings)} finding(s):",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"- {finding.format()}", file=sys.stderr)
        return 1

    print(f"Public snapshot audit passed for {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
