#!/usr/bin/env python3
"""Fail closed unless release artifacts exactly match the approved tag."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Sequence

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / ".github" / "release-readiness.json"
FINAL_VERSION_PATTERN = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
FINAL_TAG_RE = re.compile(rf"^v(?P<version>{FINAL_VERSION_PATTERN})$")
RC_TAG_RE = re.compile(
    rf"^v(?P<version>{FINAL_VERSION_PATTERN}rc(?P<number>[1-9][0-9]*))$"
)


class ReleaseArtifactError(RuntimeError):
    """Raised when a release tag or distribution artifact is inconsistent."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Name and version embedded in one distribution artifact."""

    path: Path
    name: str
    version: Version


def _load_policy(path: Path) -> dict:
    """Read one UTF-8 JSON policy object without replacing duplicate keys."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        policy = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, ValueError) as exc:
        raise ReleaseArtifactError(
            f"Could not read release policy {path}: {exc}"
        ) from exc
    if not isinstance(policy, dict):
        raise ReleaseArtifactError(f"Release policy must be a JSON object: {path}")
    return policy


def _expected_version(policy: dict, channel: str, tag: str, ref_type: str) -> Version:
    """Validate the ref and return the exact version expected in artifacts."""
    if channel not in {"pypi", "testpypi"}:
        raise ReleaseArtifactError(f"Unsupported release channel: {channel!r}.")
    if ref_type != "tag":
        raise ReleaseArtifactError(
            f"Release artifacts require a tag ref, got ref_type={ref_type!r}."
        )

    release_target = policy.get("release_target")
    if not isinstance(release_target, str) or not re.fullmatch(
        FINAL_VERSION_PATTERN, release_target
    ):
        raise ReleaseArtifactError(
            "release_target must be a canonical final X.Y.Z version."
        )
    target_version = Version(release_target)

    pattern = FINAL_TAG_RE if channel == "pypi" else RC_TAG_RE
    match = pattern.fullmatch(tag)
    expected_form = "vX.Y.Z" if channel == "pypi" else "vX.Y.ZrcN with N >= 1"
    if match is None:
        raise ReleaseArtifactError(
            f"The {channel} channel requires a canonical {expected_form} tag, got {tag!r}."
        )

    tag_version = Version(match.group("version"))
    if channel == "pypi" and tag_version != target_version:
        raise ReleaseArtifactError(
            f"Final tag version {tag_version} does not match release_target {target_version}."
        )
    if channel == "testpypi" and Version(tag_version.base_version) != target_version:
        raise ReleaseArtifactError(
            f"Release-candidate base {tag_version.base_version} does not match "
            f"release_target {target_version}."
        )
    return tag_version


def _parse_metadata(text: str, path: Path, license_identifier: str) -> ArtifactIdentity:
    """Parse identity and require the policy's licence in artifact metadata."""
    metadata = Parser().parsestr(text)
    identity_values = {}
    for field in ("Metadata-Version", "Name", "Version"):
        values = metadata.get_all(field, [])
        if len(values) != 1 or not values[0].strip():
            raise ReleaseArtifactError(
                f"Artifact metadata in {path.name} must contain exactly one "
                f"nonblank {field}."
            )
        identity_values[field] = values[0]
    name = identity_values["Name"]
    version_text = identity_values["Version"]
    expressions = metadata.get_all("License-Expression", [])
    if len(expressions) != 1 or not expressions[0].strip():
        raise ReleaseArtifactError(
            f"Artifact metadata in {path.name} must contain exactly one "
            "nonblank License-Expression."
        )
    if metadata.get_all("License", []):
        raise ReleaseArtifactError(
            f"Artifact metadata in {path.name} must not contain both License "
            "and License-Expression. Remove the legacy License field."
        )
    if expressions[0] != license_identifier:
        raise ReleaseArtifactError(
            f"Artifact metadata in {path.name} declares License-Expression "
            f"{expressions[0]!r}, which does not match license_identifier "
            f"{license_identifier!r}."
        )
    license_files = metadata.get_all("License-File", [])
    if sorted(license_files) != sorted(_REQUIRED_LICENSE_FILES):
        raise ReleaseArtifactError(
            f"Artifact metadata in {path.name} must declare each required "
            f"License-File exactly once: {sorted(_REQUIRED_LICENSE_FILES)}."
        )
    try:
        version = Version(version_text)
    except ValueError as exc:
        raise ReleaseArtifactError(
            f"Artifact metadata in {path.name} has an invalid Version."
        ) from exc
    return ArtifactIdentity(path=path, name=name, version=version)


def _wheel_identity(path: Path, license_identifier: str) -> ArtifactIdentity:
    """Return the identity embedded in a wheel METADATA member."""
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).name == "METADATA"
                and PurePosixPath(name).parent.name.endswith(".dist-info")
            ]
            if len(candidates) != 1:
                raise ReleaseArtifactError(
                    f"Wheel {path.name} must contain exactly one dist-info/METADATA."
                )
            text = archive.read(candidates[0]).decode("utf-8")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ReleaseArtifactError(
            f"Could not inspect wheel {path.name}: {exc}"
        ) from exc
    return _parse_metadata(text, path, license_identifier)


def _sdist_identity(path: Path, license_identifier: str) -> ArtifactIdentity:
    """Return the identity embedded in a source distribution PKG-INFO member."""
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            candidates = [
                member
                for member in archive.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).name == "PKG-INFO"
                and len(PurePosixPath(member.name).parts) == 2
            ]
            if len(candidates) != 1:
                raise ReleaseArtifactError(
                    f"Source distribution {path.name} must contain one top-level PKG-INFO."
                )
            stream = archive.extractfile(candidates[0])
            if stream is None:
                raise ReleaseArtifactError(f"Could not read PKG-INFO from {path.name}.")
            text = stream.read().decode("utf-8")
    except (OSError, UnicodeDecodeError, tarfile.TarError) as exc:
        raise ReleaseArtifactError(
            f"Could not inspect source distribution {path.name}: {exc}"
        ) from exc
    return _parse_metadata(text, path, license_identifier)


def _validate_unique_archive_paths(paths: Sequence[str], archive_name: str) -> None:
    """Reject unsafe paths and duplicate entries after path normalization."""
    seen = set()
    for name in paths:
        path = PurePosixPath(name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ReleaseArtifactError(f"Unsafe {archive_name} member: {name}")
        normalised = str(path)
        if normalised in seen:
            raise ReleaseArtifactError(
                f"{archive_name} contains duplicate archive member path after "
                f"normalization: {normalised!r}."
            )
        seen.add(normalised)


_REQUIRED_RUNTIME_FILES = {
    "__init__.py",
    "config/support_policy.json",
    "datasets/registry.json",
    "sequences/config/sequence_mapping.yaml",
    "vendors/config/vendor_mapping.yaml",
    "visualization/browser/viewer.js",
    "visualization/browser/viewer.css",
    "visualization/fit_monitor_page.html",
}
# The browser viewer loads every script and stylesheet of this folder.
_VIEWER_ASSET_FOLDER = PurePosixPath("visualization/browser")
_VIEWER_ASSET_SUFFIXES = {".js", ".css"}
_REQUIRED_LICENSE_FILES = {
    "LICENSE",
    "NOTICE",
    "LICENSES/brukerMRI-Apache-2.0.txt",
    "LICENSES/brukerapi-MIT.txt",
}
_SDIST_ROOT_FILES = {
    *_REQUIRED_LICENSE_FILES,
    "README.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "MAINTAINERS.md",
    "SECURITY.md",
    "SUPPORT.md",
    "pyproject.toml",
    "MANIFEST.in",
    "PKG-INFO",
    "setup.cfg",
}
_DIST_INFO_FILES = {
    "METADATA",
    "WHEEL",
    "RECORD",
    "entry_points.txt",
    "top_level.txt",
} | {f"licenses/{path}" for path in _REQUIRED_LICENSE_FILES}


def _check_runtime_file(relative: str, member: str, label: str) -> None:
    """Reject a packaged file that is neither Python source nor a known asset.

    Parameters
    ----------
    relative : str
        Path of the file relative to the ``hypermri`` package.
    member : str
        Archive member name, used in the error message.
    label : str
        ``"wheel"`` or ``"source distribution"``, used in the error message.

    Raises
    ------
    ReleaseArtifactError
        If the file is not Python source, a required configuration or viewer
        file, or a script or stylesheet of the browser viewer.
    """
    path = PurePosixPath(relative)
    if path.suffix == ".py" or relative in _REQUIRED_RUNTIME_FILES:
        return
    if path.parent == _VIEWER_ASSET_FOLDER and path.suffix in _VIEWER_ASSET_SUFFIXES:
        return
    if path.suffix in {".json", ".yaml"}:
        raise ReleaseArtifactError(f"Unexpected runtime configuration: {member}")
    raise ReleaseArtifactError(f"Unexpected runtime file in {label}: {member}")


def _validate_distribution_contents(wheel: Path, sdist: Path) -> None:
    """Check the runtime-only distribution boundary without extracting files."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            wheel_infos = archive.infolist()
            _validate_unique_archive_paths(
                [info.filename for info in wheel_infos], f"Wheel {wheel.name}"
            )
            wheel_members = [info.filename for info in wheel_infos if not info.is_dir()]
            dist_info_directories = {
                PurePosixPath(name).parts[0]
                for name in wheel_members
                if PurePosixPath(name).parts[0].endswith(".dist-info")
            }
            if len(dist_info_directories) != 1:
                raise ReleaseArtifactError(
                    f"Wheel {wheel.name} must contain exactly one dist-info directory."
                )
            wheel_runtime = {}
            wheel_licenses = {}
            for name in wheel_members:
                path = PurePosixPath(name)
                if name.startswith("hypermri/"):
                    relative = name.removeprefix("hypermri/")
                    _check_runtime_file(relative, name, "wheel")
                    wheel_runtime[relative] = archive.read(name)
                    continue
                if not (
                    path.parts[0].endswith(".dist-info")
                    and str(PurePosixPath(*path.parts[1:])) in _DIST_INFO_FILES
                ):
                    raise ReleaseArtifactError(f"Unexpected wheel member: {name}")
                relative = str(PurePosixPath(*path.parts[1:]))
                if relative.startswith("licenses/"):
                    license_file = relative.removeprefix("licenses/")
                    if license_file in _REQUIRED_LICENSE_FILES:
                        if license_file in wheel_licenses:
                            raise ReleaseArtifactError(
                                f"Wheel contains duplicate license file {license_file}."
                            )
                        wheel_licenses[license_file] = archive.read(name)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseArtifactError(
            f"Could not inspect wheel {wheel.name}: {exc}"
        ) from exc

    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ReleaseArtifactError(
                    "Source distribution contains a link or special file."
                )
            _validate_unique_archive_paths(
                [member.name for member in members],
                f"Source distribution {sdist.name}",
            )
            roots = {PurePosixPath(member.name).parts[0] for member in members}
            if len(roots) != 1:
                raise ReleaseArtifactError(
                    "Source distribution must have one root directory."
                )
            sdist_members = [
                str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
                for member in members
                if member.isfile()
            ]
            sdist_runtime = {}
            sdist_licenses = {}
            for member in members:
                if not member.isfile():
                    continue
                relative = str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
                path = PurePosixPath(relative)
                if relative.startswith("src/hypermri/"):
                    runtime_file = relative.removeprefix("src/hypermri/")
                    _check_runtime_file(runtime_file, relative, "source distribution")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ReleaseArtifactError(
                            f"Could not read runtime file {relative} from {sdist.name}."
                        )
                    sdist_runtime[runtime_file] = stream.read()
                elif relative in _REQUIRED_LICENSE_FILES:
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ReleaseArtifactError(
                            f"Could not read license file {relative} from {sdist.name}."
                        )
                    sdist_licenses[relative] = stream.read()
                elif not (
                    relative in _SDIST_ROOT_FILES
                    or relative == "docs/source/development/release.rst"
                    or (
                        len(path.parts) == 3
                        and path.parts[:2] == ("src", "hypermri.egg-info")
                    )
                ):
                    raise ReleaseArtifactError(
                        f"Unexpected source distribution member: {relative}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseArtifactError(
            f"Could not inspect source distribution {sdist.name}: {exc}"
        ) from exc
    missing = _REQUIRED_RUNTIME_FILES - wheel_runtime.keys()
    if missing:
        raise ReleaseArtifactError(f"Wheel is missing runtime files: {sorted(missing)}")
    if sdist_runtime.keys() != wheel_runtime.keys():
        raise ReleaseArtifactError(
            "Wheel and source distribution runtime file sets differ."
        )
    different_files = sorted(
        name for name in wheel_runtime if wheel_runtime[name] != sdist_runtime[name]
    )
    if different_files:
        raise ReleaseArtifactError(
            "Wheel and source distribution runtime contents differ for: "
            f"{different_files}."
        )
    for license_file in sorted(_REQUIRED_LICENSE_FILES):
        for label, contents in (
            ("Wheel", wheel_licenses),
            ("Source distribution", sdist_licenses),
        ):
            if license_file not in contents:
                raise ReleaseArtifactError(
                    f"{label} is missing required license file {license_file}."
                )
            if not contents[license_file].strip():
                raise ReleaseArtifactError(
                    f"{label} contains an empty license file {license_file}."
                )
        if wheel_licenses[license_file] != sdist_licenses[license_file]:
            raise ReleaseArtifactError(
                f"Wheel/source distribution license contents differ for {license_file}."
            )
    for required in (
        "pyproject.toml",
        "docs/source/development/release.rst",
    ):
        if required not in sdist_members:
            raise ReleaseArtifactError(f"Source distribution is missing {required}.")


def validate_release_artifacts(
    *,
    channel: str,
    tag: str,
    ref_type: str,
    dist_dir: Path,
    policy_path: Path = POLICY_PATH,
) -> Version:
    """Validate release identity, licence and runtime distribution contents.

    Parameters
    ----------
    channel : str
        ``pypi`` for a final release or ``testpypi`` for a release candidate.
    tag : str
        Canonical release tag matching the approved target in the policy.
    ref_type : str
        Must be ``tag``; branch refs cannot identify a release.
    dist_dir : pathlib.Path
        Directory containing exactly one wheel and one source archive.
    policy_path : pathlib.Path, optional
        UTF-8 JSON policy object owning the distribution name, target version
        and exact licence expression in ``license_identifier``.
        Duplicate keys in any nested object, including escaped spellings of the
        same key, are rejected. Relative paths use the current working directory.

    Returns
    -------
    packaging.version.Version
        Approved version after filenames, embedded identity/licence metadata
        and contents agree with the policy.
        Validation reads archives eagerly without extracting or changing files.

    Raises
    ------
    ReleaseArtifactError
        If the policy cannot be read as a UTF-8 JSON object or contains duplicate
        keys, if artifact identity metadata does not contain exactly one nonblank
        ``Metadata-Version``, ``Name`` and ``Version``, or if identity or
        contents violate the release policy. Each artifact must contain exactly
        one nonblank ``License-Expression`` matching ``license_identifier`` and
        must omit the legacy ``License`` header. This includes missing runtime
        configuration, unsafe or duplicate normalized member paths, unexpected
        repository/data files or unequal runtime paths or bytes between the wheel
        and source archive. All four required licence/notice files must be
        nonempty, declared once in each artifact's ``License-File`` metadata and
        have identical bytes in the wheel and source archive.
    """
    policy = _load_policy(policy_path)
    expected_version = _expected_version(policy, channel, tag, ref_type)
    distribution_name = policy.get("distribution_name")
    if not isinstance(distribution_name, str) or not distribution_name.strip():
        raise ReleaseArtifactError(
            "release policy must define a non-empty distribution_name."
        )
    expected_name = canonicalize_name(distribution_name)
    license_identifier = policy.get("license_identifier")
    if not isinstance(license_identifier, str) or not license_identifier.strip():
        raise ReleaseArtifactError(
            "release policy must define a non-empty license_identifier."
        )

    if not dist_dir.is_dir():
        raise ReleaseArtifactError(f"Distribution directory does not exist: {dist_dir}")
    entries = sorted(dist_dir.iterdir())
    wheels = [path for path in entries if path.is_file() and path.suffix == ".whl"]
    sdists = [
        path for path in entries if path.is_file() and path.name.endswith(".tar.gz")
    ]
    if len(entries) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseArtifactError(
            "dist must contain exactly one wheel and one .tar.gz source distribution."
        )

    try:
        wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
        sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
    except ValueError as exc:
        raise ReleaseArtifactError(f"Invalid distribution filename: {exc}") from exc

    filename_identities = (
        ArtifactIdentity(wheels[0], wheel_name, wheel_version),
        ArtifactIdentity(sdists[0], sdist_name, sdist_version),
    )
    _validate_distribution_contents(wheels[0], sdists[0])
    embedded_identities = (
        _wheel_identity(wheels[0], license_identifier),
        _sdist_identity(sdists[0], license_identifier),
    )
    for identity in (*filename_identities, *embedded_identities):
        if canonicalize_name(identity.name) != expected_name:
            raise ReleaseArtifactError(
                f"Artifact {identity.path.name} identifies distribution {identity.name!r}, "
                f"expected {distribution_name!r}."
            )
        if identity.version != expected_version:
            raise ReleaseArtifactError(
                f"Artifact {identity.path.name} identifies version {identity.version}, "
                f"expected {expected_version}."
            )

    return expected_version


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate release refs and built distribution artifacts."
    )
    parser.add_argument("--channel", choices=("pypi", "testpypi"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ref-type", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict artifact validator."""
    args = build_parser().parse_args(argv)
    try:
        version = validate_release_artifacts(
            channel=args.channel,
            tag=args.tag,
            ref_type=args.ref_type,
            dist_dir=args.dist_dir,
            policy_path=args.policy,
        )
    except ReleaseArtifactError as exc:
        print(f"Release artifact validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {args.channel} artifacts for hypermri {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
