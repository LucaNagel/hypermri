"""Tests for fail-closed release artifact validation."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from packaging.version import Version


def _load_validator_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate_release_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypermri_validate_release_artifacts", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator_script()
_LICENSE_CONTENTS = {
    "LICENSE": b"invented project licence",
    "NOTICE": b"invented third-party attribution",
    "LICENSES/brukerMRI-Apache-2.0.txt": b"invented upstream licence",
    "LICENSES/brukerapi-MIT.txt": b"invented upstream MIT licence",
}


def _write_policy(tmp_path: Path, target: str = "0.1.0") -> Path:
    policy_path = tmp_path / "release-readiness.json"
    policy_path.write_text(
        json.dumps(
            {
                "release_target": target,
                "distribution_name": "hypermri",
                "license_identifier": "AGPL-3.0-only",
            }
        ),
        encoding="utf-8",
    )
    return policy_path


def _metadata(name: str, version: str) -> bytes:
    license_headers = "".join(f"License-File: {path}\n" for path in _LICENSE_CONTENTS)
    return (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        f"License-Expression: AGPL-3.0-only\n{license_headers}"
    ).encode()


def _metadata_with_headers(
    headers: str,
    description: str = "",
    *,
    license_expression_headers: str = "License-Expression: AGPL-3.0-only\n",
) -> bytes:
    license_headers = "".join(f"License-File: {path}\n" for path in _LICENSE_CONTENTS)
    return (
        f"{headers}{license_expression_headers}{license_headers}\n{description}"
    ).encode()


def _replace_metadata(dist: Path, artifact: str, payload: bytes) -> None:
    member = (
        "hypermri-0.1.0.dist-info/METADATA"
        if artifact == "wheel"
        else "hypermri-0.1.0/PKG-INFO"
    )
    _rewrite_member(dist, artifact, member, payload)


def _validate_test_artifacts(dist: Path, tmp_path: Path) -> Version:
    return validator.validate_release_artifacts(
        channel="pypi",
        tag="v0.1.0",
        ref_type="tag",
        dist_dir=dist,
        policy_path=_write_policy(tmp_path),
    )


def _write_artifacts(
    tmp_path: Path,
    filename_version: str,
    *,
    metadata_version: str | None = None,
    extra_wheel: str | None = None,
    extra_sdist: str | None = None,
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    embedded_version = metadata_version or filename_version
    wheel = dist / f"hypermri-{filename_version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        for name in validator._REQUIRED_RUNTIME_FILES:
            archive.writestr(f"hypermri/{name}", b"invented source/configuration")
        if extra_wheel:
            archive.writestr(extra_wheel, b"invented excluded content")
        archive.writestr(
            f"hypermri-{filename_version}.dist-info/METADATA",
            _metadata("hypermri", embedded_version),
        )
        for name, content in _LICENSE_CONTENTS.items():
            archive.writestr(
                f"hypermri-{filename_version}.dist-info/licenses/{name}", content
            )

    sdist = dist / f"hypermri-{filename_version}.tar.gz"
    payload = _metadata("hypermri", embedded_version)
    info = tarfile.TarInfo(f"hypermri-{filename_version}/PKG-INFO")
    info.size = len(payload)
    with tarfile.open(sdist, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
        names = {f"src/hypermri/{name}" for name in validator._REQUIRED_RUNTIME_FILES}
        names.update(
            {
                "pyproject.toml",
                "docs/source/development/release.rst",
                *_LICENSE_CONTENTS,
            }
        )
        if extra_sdist:
            names.add(extra_sdist)
        for name in names:
            content = _LICENSE_CONTENTS.get(name, b"invented source/configuration")
            entry = tarfile.TarInfo(f"hypermri-{filename_version}/{name}")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return dist


def _rewrite_member(
    dist: Path, artifact: str, name: str, payload: bytes | None
) -> None:
    rewritten = io.BytesIO()
    if artifact == "wheel":
        path = next(dist.glob("*.whl"))
        with (
            zipfile.ZipFile(path) as original,
            zipfile.ZipFile(rewritten, mode="w") as archive,
        ):
            for member in original.infolist():
                content = original.read(member)
                if member.filename == name:
                    if payload is None:
                        continue
                    content = payload
                archive.writestr(member, content)
    else:
        path = next(dist.glob("*.tar.gz"))
        with (
            tarfile.open(path, mode="r:gz") as original,
            tarfile.open(fileobj=rewritten, mode="w:gz") as archive,
        ):
            for member in original.getmembers():
                content = original.extractfile(member) if member.isfile() else None
                if member.name == name:
                    if payload is None:
                        continue
                    member.size = len(payload)
                    content = io.BytesIO(payload)
                archive.addfile(member, content)
    path.write_bytes(rewritten.getvalue())


def _append_wheel_member(dist: Path, name: str) -> None:
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel, mode="a") as archive:
        archive.writestr(name, b"duplicate archive member")


def _append_sdist_member(dist: Path, name: str) -> None:
    sdist = next(dist.glob("*.tar.gz"))
    root = sdist.name.removesuffix(".tar.gz")
    payload = b"duplicate archive member"
    info = tarfile.TarInfo(f"{root}/{name}")
    info.size = len(payload)
    rewritten = io.BytesIO()
    with (
        tarfile.open(sdist, mode="r:gz") as original,
        tarfile.open(fileobj=rewritten, mode="w:gz") as archive,
    ):
        for member in original.getmembers():
            content = original.extractfile(member) if member.isfile() else None
            archive.addfile(member, content)
        archive.addfile(info, io.BytesIO(payload))
    sdist.write_bytes(rewritten.getvalue())


def _corrupt_distribution(dist: Path, pattern: str) -> Path:
    archive = next(dist.glob(pattern))
    archive.write_bytes(b"not a valid distribution archive")
    return archive


@pytest.mark.parametrize(
    "channel,tag,version",
    [
        ("pypi", "v0.1.0", "0.1.0"),
        ("testpypi", "v0.1.0rc1", "0.1.0rc1"),
    ],
)
def test_valid_release_artifacts(channel, tag, version, tmp_path: Path):
    dist = _write_artifacts(tmp_path, version)

    result = validator.validate_release_artifacts(
        channel=channel,
        tag=tag,
        ref_type="tag",
        dist_dir=dist,
        policy_path=_write_policy(tmp_path),
    )

    assert result == Version(version)


@pytest.mark.parametrize(
    "channel,version", [("pypi", "0.1.0"), ("testpypi", "0.1.0rc1")]
)
@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "license_headers,expected",
    [
        ("", "exactly one nonblank License-Expression"),
        ("License-Expression:\n", "exactly one nonblank License-Expression"),
        ("License-Expression:   \n", "exactly one nonblank License-Expression"),
        ("License-Expression: MIT\n", "does not match license_identifier"),
        (
            "License-Expression: AGPL-3.0-only OR MIT\n",
            "does not match license_identifier",
        ),
        (
            "License-Expression: AGPL-3.0-only\nlicense-expression: AGPL-3.0-only\n",
            "exactly one nonblank License-Expression",
        ),
        (
            "License-Expression: AGPL-3.0-only\nLICENSE-EXPRESSION: MIT\n",
            "exactly one nonblank License-Expression",
        ),
        (
            "License-Expression: AGPL-3.0-only\nLicense: AGPL-3.0-only\n",
            "must not contain both License and License-Expression",
        ),
        (
            "License-Expression: AGPL-3.0-only\nlicense: MIT\n",
            "must not contain both License and License-Expression",
        ),
    ],
    ids=[
        "missing",
        "empty",
        "blank",
        "different",
        "broader",
        "duplicate-identical",
        "duplicate-conflicting",
        "legacy-identical",
        "legacy-conflicting",
    ],
)
def test_artifact_license_declaration_must_match_policy(
    tmp_path, capsys, channel, version, artifact, license_headers, expected
):
    """Both publishing channels reject unapproved or ambiguous artifact licences."""
    dist = _write_artifacts(tmp_path, version)
    member = (
        f"hypermri-{version}.dist-info/METADATA"
        if artifact == "wheel"
        else f"hypermri-{version}/PKG-INFO"
    )
    payload = _metadata_with_headers(
        f"Metadata-Version: 2.4\nName: hypermri\nVersion: {version}\n",
        license_expression_headers=license_headers,
    )
    _rewrite_member(dist, artifact, member, payload)
    policy = _write_policy(tmp_path)

    with pytest.raises(validator.ReleaseArtifactError, match=expected) as error:
        validator.validate_release_artifacts(
            channel=channel,
            tag=f"v{version}",
            ref_type="tag",
            dist_dir=dist,
            policy_path=policy,
        )
    affected = next(dist.glob("*.whl" if artifact == "wheel" else "*.tar.gz"))
    assert affected.name in str(error.value)
    assert (
        validator.main(
            [
                "--channel",
                channel,
                "--tag",
                f"v{version}",
                "--ref-type",
                "tag",
                "--dist-dir",
                str(dist),
                "--policy",
                str(policy),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert expected in output.err and affected.name in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_license_header_names_are_case_insensitive_and_description_is_ignored(
    tmp_path, artifact
):
    """Only metadata headers declare the licence; README prose is not a header."""
    dist = _write_artifacts(tmp_path, "0.1.0")
    payload = _metadata_with_headers(
        "Metadata-Version: 2.4\nName: hypermri\nVersion: 0.1.0\n",
        "License-Expression: MIT\nLicense: MIT\n",
        license_expression_headers="lIcEnSe-ExPrEsSiOn: AGPL-3.0-only\n",
    )
    _replace_metadata(dist, artifact, payload)

    assert _validate_test_artifacts(dist, tmp_path) == Version("0.1.0")


@pytest.mark.parametrize("value", [None, "", "   ", False, 1, [], {}])
def test_artifact_policy_requires_a_nonblank_license_identifier(tmp_path, value):
    """An unset or incorrectly typed policy licence cannot permit release."""
    dist = _write_artifacts(tmp_path, "0.1.0")
    policy = _write_policy(tmp_path)
    contents = json.loads(policy.read_text(encoding="utf-8"))
    if value is None:
        del contents["license_identifier"]
    else:
        contents["license_identifier"] = value
    policy.write_text(json.dumps(contents), encoding="utf-8")

    with pytest.raises(validator.ReleaseArtifactError, match="license_identifier"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=policy,
        )


@pytest.mark.parametrize(
    "channel,version", [("pypi", "0.1.0"), ("testpypi", "0.1.0rc1")]
)
@pytest.mark.parametrize(
    "extra_fields,key",
    [
        ('"release_target": "0.2.0",', "release_target"),
        ('"release_target": "0.1.0",', "release_target"),
        ('"distribution_name": "another-project",', "distribution_name"),
        (r'"release_\u0074arget": "0.2.0",', "release_target"),
        ('"review": {"approved": false, "approved": true},', "approved"),
        ('"reviews": [{"approved": false, "approved": true}],', "approved"),
    ],
    ids=[
        "conflicting-target",
        "identical-target",
        "conflicting-name",
        "escaped-target",
        "nested-object",
        "object-in-array",
    ],
)
def test_duplicate_policy_keys_fail_closed_in_api_and_cli(
    tmp_path, capsys, channel, version, extra_fields, key
):
    """Repeated decisions cannot be replaced silently in either release channel."""
    dist = _write_artifacts(tmp_path, version)
    policy = _write_policy(tmp_path)
    policy.write_text(
        "{"
        + extra_fields
        + '"release_target": "0.1.0", "distribution_name": "hypermri"}',
        encoding="utf-8",
    )
    expected = f"Duplicate JSON key {key!r}"

    with pytest.raises(validator.ReleaseArtifactError) as error:
        validator.validate_release_artifacts(
            channel=channel,
            tag=f"v{version}",
            ref_type="tag",
            dist_dir=dist,
            policy_path=policy,
        )
    assert expected in str(error.value)
    assert str(policy) in str(error.value)
    result = validator.main(
        [
            "--channel",
            channel,
            "--tag",
            f"v{version}",
            "--ref-type",
            "tag",
            "--dist-dir",
            str(dist),
            "--policy",
            str(policy),
        ]
    )
    output = capsys.readouterr()
    assert result == 1
    assert expected in output.err and str(policy) in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "channel,version", [("pypi", "0.1.0"), ("testpypi", "0.1.0rc1")]
)
@pytest.mark.parametrize(
    "contents,expected",
    [
        (b"\xff", "Could not read release policy"),
        (b"{", "Could not read release policy"),
        (None, "Could not read release policy"),
        (b"[]", "Release policy must be a JSON object"),
    ],
    ids=["invalid-utf8", "malformed-json", "missing-file", "non-object"],
)
def test_unreadable_or_invalid_policy_fails_closed_in_api_and_cli(
    tmp_path, capsys, channel, version, contents, expected
):
    """Policy-file failures identify their path without an uncaught traceback."""
    dist = _write_artifacts(tmp_path, version)
    policy = tmp_path / "release-readiness.json"
    if contents is not None:
        policy.write_bytes(contents)

    with pytest.raises(validator.ReleaseArtifactError, match=expected) as error:
        validator.validate_release_artifacts(
            channel=channel,
            tag=f"v{version}",
            ref_type="tag",
            dist_dir=dist,
            policy_path=policy,
        )
    assert str(policy) in str(error.value)
    result = validator.main(
        [
            "--channel",
            channel,
            "--tag",
            f"v{version}",
            "--ref-type",
            "tag",
            "--dist-dir",
            str(dist),
            "--policy",
            str(policy),
        ]
    )
    output = capsys.readouterr()
    assert result == 1
    assert expected in output.err and str(policy) in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize(
    "channel,version", [("pypi", "0.1.0"), ("testpypi", "0.1.0rc1")]
)
def test_policy_keys_in_distinct_objects_remain_valid(tmp_path, channel, version):
    """The same field name in distinct objects is unambiguous."""
    dist = _write_artifacts(tmp_path, version)
    policy = _write_policy(tmp_path)
    contents = json.loads(policy.read_text(encoding="utf-8"))
    contents["reviews"] = [{"approved": False}, {"approved": True}]
    policy.write_text(json.dumps(contents), encoding="utf-8")

    assert validator.validate_release_artifacts(
        channel=channel,
        tag=f"v{version}",
        ref_type="tag",
        dist_dir=dist,
        policy_path=policy,
    ) == Version(version)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "header,alias,original,conflicting",
    [
        ("Metadata-Version", "metadata-version", "2.4", "2.3"),
        ("Name", "nAmE", "hypermri", "another-project"),
        ("Version", "vErSiOn", "0.1.0", "0.2.0"),
    ],
)
@pytest.mark.parametrize("duplicate_value", ["identical", "conflicting"])
def test_release_artifacts_reject_duplicate_identity_headers(
    tmp_path: Path,
    artifact: str,
    header: str,
    alias: str,
    original: str,
    conflicting: str,
    duplicate_value: str,
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    repeated = original if duplicate_value == "identical" else conflicting
    fields = {
        "Metadata-Version": "2.4",
        "Name": "hypermri",
        "Version": "0.1.0",
    }
    del fields[header]
    headers = "".join(f"{name}: {value}\n" for name, value in fields.items())
    payload = _metadata_with_headers(
        f"{headers}{header}: {original}\n{alias}: {repeated}\n"
    )
    _replace_metadata(dist, artifact, payload)

    with pytest.raises(
        validator.ReleaseArtifactError, match=rf"exactly one nonblank {header}"
    ):
        _validate_test_artifacts(dist, tmp_path)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("header", ["Metadata-Version", "Name", "Version"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_release_artifacts_require_one_nonblank_identity_header(
    tmp_path: Path, artifact: str, header: str, value: str | None
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    fields = {
        "Metadata-Version": "2.4",
        "Name": "hypermri",
        "Version": "0.1.0",
    }
    if value is None:
        del fields[header]
    else:
        fields[header] = value
    headers = "".join(
        f"{name}: {field_value}\n" for name, field_value in fields.items()
    )
    _replace_metadata(dist, artifact, _metadata_with_headers(headers))

    with pytest.raises(
        validator.ReleaseArtifactError,
        match=rf"exactly one nonblank {header}",
    ):
        _validate_test_artifacts(dist, tmp_path)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_release_artifacts_accept_case_insensitive_identity_headers_and_description(
    tmp_path: Path, artifact: str
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    payload = _metadata_with_headers(
        "mEtAdAtA-vErSiOn: 2.4\nnAmE: hypermri\nvErSiOn: 0.1.0\n",
        "Metadata-Version: 9.9\nName: body text\nVersion: 9.9\n",
    )
    _replace_metadata(dist, artifact, payload)

    assert _validate_test_artifacts(dist, tmp_path) == Version("0.1.0")


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "runtime_file",
    [
        "__init__.py",
        "analysis/invented.py",
        "config/support_policy.json",
        "sequences/config/sequence_mapping.yaml",
    ],
)
def test_runtime_bytes_must_match_between_artifacts(tmp_path, artifact, runtime_file):
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        extra_wheel="hypermri/analysis/invented.py",
        extra_sdist="src/hypermri/analysis/invented.py",
    )
    for current_artifact, prefix in (
        ("wheel", "hypermri/"),
        ("sdist", "hypermri-0.1.0/src/hypermri/"),
    ):
        _rewrite_member(dist, current_artifact, prefix + runtime_file, b"original")
    prefix = "hypermri/" if artifact == "wheel" else "hypermri-0.1.0/src/hypermri/"
    _rewrite_member(dist, artifact, prefix + runtime_file, b"modified")

    with pytest.raises(
        validator.ReleaseArtifactError, match="runtime contents differ"
    ) as error:
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )
    assert runtime_file in str(error.value)


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
def test_runtime_paths_must_match_between_artifacts(tmp_path, artifact):
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        extra_wheel="hypermri/analysis/invented.py" if artifact == "wheel" else None,
        extra_sdist="src/hypermri/analysis/invented.py"
        if artifact == "sdist"
        else None,
    )

    with pytest.raises(
        validator.ReleaseArtifactError, match="runtime file sets differ"
    ):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_browser_viewer_assets_are_accepted(tmp_path):
    """Regression: the viewer's scripts and stylesheets used to block releases.

    The validator accepted only ``.py``, ``.json`` and ``.yaml`` files, so a
    real ``0.1.0rc1`` wheel failed on ``visualization/fit_monitor_page.html``
    and the scripts in ``visualization/browser``.
    """
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        extra_wheel="hypermri/visualization/browser/image_panel.js",
        extra_sdist="src/hypermri/visualization/browser/image_panel.js",
    )
    for artifact, prefix in (
        ("wheel", "hypermri/"),
        ("sdist", "hypermri-0.1.0/src/hypermri/"),
    ):
        _rewrite_member(
            dist, artifact, prefix + "visualization/browser/image_panel.js", b"// js"
        )

    validator.validate_release_artifacts(
        channel="pypi",
        tag="v0.1.0",
        ref_type="tag",
        dist_dir=dist,
        policy_path=_write_policy(tmp_path),
    )


@pytest.mark.parametrize(
    "runtime_file",
    [
        "visualization/browser/viewer.js",
        "visualization/browser/viewer.css",
        "visualization/fit_monitor_page.html",
    ],
)
def test_viewer_entry_files_are_required(tmp_path, monkeypatch, runtime_file):
    """A wheel without the viewer's entry files cannot show any view."""
    required = validator._REQUIRED_RUNTIME_FILES
    monkeypatch.setattr(validator, "_REQUIRED_RUNTIME_FILES", required - {runtime_file})
    dist = _write_artifacts(tmp_path, "0.1.0")
    monkeypatch.setattr(validator, "_REQUIRED_RUNTIME_FILES", required)

    with pytest.raises(validator.ReleaseArtifactError, match="missing runtime files"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "runtime_file",
    [
        "visualization/browser/notes.html",
        "visualization/browser/nested/extra.js",
        "visualization/extra.js",
        "analysis/table.csv",
    ],
)
def test_other_non_python_runtime_files_stay_rejected(tmp_path, runtime_file):
    """Only the viewer folder's scripts and stylesheets widen the boundary."""
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        extra_wheel=f"hypermri/{runtime_file}",
        extra_sdist=f"src/hypermri/{runtime_file}",
    )

    with pytest.raises(
        validator.ReleaseArtifactError, match="Unexpected runtime file in wheel"
    ):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("payload", [b"", b"identical changed runtime source"])
def test_matching_runtime_bytes_are_accepted(tmp_path, payload):
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        extra_wheel="hypermri/analysis/invented.py",
        extra_sdist="src/hypermri/analysis/invented.py",
    )
    for artifact, prefix in (
        ("wheel", "hypermri/"),
        ("sdist", "hypermri-0.1.0/src/hypermri/"),
    ):
        _rewrite_member(dist, artifact, prefix + "analysis/invented.py", payload)

    assert validator.validate_release_artifacts(
        channel="pypi",
        tag="v0.1.0",
        ref_type="tag",
        dist_dir=dist,
        policy_path=_write_policy(tmp_path),
    ) == Version("0.1.0")


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("license_file", _LICENSE_CONTENTS)
def test_release_artifacts_require_each_license_file(tmp_path, artifact, license_file):
    dist = _write_artifacts(tmp_path, "0.1.0")
    member = (
        f"hypermri-0.1.0.dist-info/licenses/{license_file}"
        if artifact == "wheel"
        else f"hypermri-0.1.0/{license_file}"
    )
    _rewrite_member(dist, artifact, member, None)

    with pytest.raises(validator.ReleaseArtifactError, match="missing.*licen[sc]e"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("license_file", _LICENSE_CONTENTS)
def test_license_metadata_must_declare_each_file(tmp_path, artifact, license_file):
    dist = _write_artifacts(tmp_path, "0.1.0")
    member = (
        "hypermri-0.1.0.dist-info/METADATA"
        if artifact == "wheel"
        else "hypermri-0.1.0/PKG-INFO"
    )
    payload = _metadata("hypermri", "0.1.0").replace(
        f"License-File: {license_file}\n".encode(), b""
    )
    _rewrite_member(dist, artifact, member, payload)

    with pytest.raises(validator.ReleaseArtifactError, match="License-File"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("license_file", ["NOTICE", "unreviewed.txt"])
def test_license_metadata_rejects_duplicate_or_unreviewed_files(
    tmp_path, artifact, license_file
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    member = (
        "hypermri-0.1.0.dist-info/METADATA"
        if artifact == "wheel"
        else "hypermri-0.1.0/PKG-INFO"
    )
    payload = (
        _metadata("hypermri", "0.1.0") + f"License-File: {license_file}\n".encode()
    )
    _rewrite_member(dist, artifact, member, payload)

    with pytest.raises(validator.ReleaseArtifactError, match="License-File"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("relocate_all", [False, True])
def test_wheel_licenses_belong_to_the_metadata_distribution(tmp_path, relocate_all):
    dist = _write_artifacts(tmp_path, "0.1.0")
    wheel = next(dist.glob("*.whl"))
    rewritten = io.BytesIO()
    prefix = "hypermri-0.1.0.dist-info/licenses/"
    with (
        zipfile.ZipFile(wheel) as original,
        zipfile.ZipFile(rewritten, mode="w") as archive,
    ):
        for member in original.infolist():
            name = member.filename
            if name.startswith(prefix) and (relocate_all or name == prefix + "NOTICE"):
                name = "other.dist-info/licenses/" + name.removeprefix(prefix)
            archive.writestr(name, original.read(member))
    wheel.write_bytes(rewritten.getvalue())

    with pytest.raises(validator.ReleaseArtifactError, match="dist-info"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("license_file", _LICENSE_CONTENTS)
@pytest.mark.parametrize("payload", [b"changed licence or attribution", b""])
def test_licenses_must_be_nonempty_and_identical_between_artifacts(
    tmp_path, artifact, license_file, payload
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    member = (
        f"hypermri-0.1.0.dist-info/licenses/{license_file}"
        if artifact == "wheel"
        else f"hypermri-0.1.0/{license_file}"
    )
    _rewrite_member(dist, artifact, member, payload)

    with pytest.raises(
        validator.ReleaseArtifactError,
        match="license contents differ|empty license file",
    ):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_release_artifacts_reject_branch_ref(tmp_path: Path):
    dist = _write_artifacts(tmp_path, "0.1.0rc1")

    with pytest.raises(validator.ReleaseArtifactError, match="tag ref"):
        validator.validate_release_artifacts(
            channel="testpypi",
            tag="main",
            ref_type="branch",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "channel,tag,version",
    [
        ("pypi", "v0.1.0rc1", "0.1.0rc1"),
        ("testpypi", "v0.1.0", "0.1.0"),
        ("testpypi", "v0.1.0rc0", "0.1.0rc0"),
        ("testpypi", "v0.1.0-rc1", "0.1.0rc1"),
    ],
)
def test_release_artifacts_reject_noncanonical_channel_tags(
    channel,
    tag,
    version,
    tmp_path: Path,
):
    dist = _write_artifacts(tmp_path, version)

    with pytest.raises(validator.ReleaseArtifactError, match="canonical"):
        validator.validate_release_artifacts(
            channel=channel,
            tag=tag,
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_release_artifacts_reject_wrong_release_target(tmp_path: Path):
    dist = _write_artifacts(tmp_path, "0.2.0")

    with pytest.raises(validator.ReleaseArtifactError, match="release_target"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.2.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_release_artifacts_reject_embedded_metadata_mismatch(tmp_path: Path):
    dist = _write_artifacts(
        tmp_path,
        "0.1.0",
        metadata_version="0.2.0",
    )

    with pytest.raises(validator.ReleaseArtifactError, match="version 0.2.0"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_release_artifacts_reject_extra_dist_entry(tmp_path: Path):
    dist = _write_artifacts(tmp_path, "0.1.0")
    (dist / "checksums.txt").write_text("unreviewed", encoding="utf-8")

    with pytest.raises(validator.ReleaseArtifactError, match="exactly one wheel"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "member",
    [
        ".gitlab-ci.yml",
        ".lfsconfig",
        "tests/invented.txt",
        "examples/fid",
        "docs/source/tutorials/notebooks/invented.ipynb",
        "src/hypermri/raw.npy",
        "src/hypermri/config/unowned.json",
    ],
)
def test_source_distribution_rejects_repository_and_data_files(tmp_path, member):
    dist = _write_artifacts(tmp_path, "0.1.0", extra_sdist=member)
    with pytest.raises(validator.ReleaseArtifactError, match="Unexpected"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize("member", ["tests/invented.py", "hypermri/raw.npy"])
def test_wheel_rejects_test_and_data_files(tmp_path, member):
    dist = _write_artifacts(tmp_path, "0.1.0", extra_wheel=member)
    with pytest.raises(validator.ReleaseArtifactError, match="Unexpected"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "duplicate_member",
    [
        "hypermri/__init__.py",
        "hypermri/./__init__.py",
        "hypermri-0.1.0.dist-info/METADATA",
    ],
)
def test_release_artifacts_reject_duplicate_normalized_wheel_members(
    tmp_path: Path, duplicate_member: str
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    _append_wheel_member(dist, duplicate_member)

    with pytest.raises(
        validator.ReleaseArtifactError, match="duplicate archive member"
    ):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "duplicate_member",
    [
        "src/hypermri/__init__.py",
        "src/hypermri/./__init__.py",
        "LICENSE",
    ],
)
def test_release_artifacts_reject_duplicate_normalized_sdist_members(
    tmp_path: Path, duplicate_member: str
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    _append_sdist_member(dist, duplicate_member)

    with pytest.raises(
        validator.ReleaseArtifactError, match="duplicate archive member"
    ):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


def test_unique_wheel_alias_does_not_relax_runtime_boundary(tmp_path: Path):
    """Normalization detects collisions without admitting new runtime paths."""
    dist = _write_artifacts(tmp_path, "0.1.0")
    wheel = next(dist.glob("*.whl"))
    rewritten = io.BytesIO()
    with (
        zipfile.ZipFile(wheel) as original,
        zipfile.ZipFile(rewritten, mode="w") as archive,
    ):
        for member in original.infolist():
            name = member.filename
            if name == "hypermri/__init__.py":
                name = "hypermri/./__init__.py"
            archive.writestr(name, original.read(member))
    wheel.write_bytes(rewritten.getvalue())

    with pytest.raises(validator.ReleaseArtifactError, match="runtime"):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=_write_policy(tmp_path),
        )


@pytest.mark.parametrize(
    "pattern,archive_label",
    [
        ("*.whl", "wheel"),
        ("*.tar.gz", "source distribution"),
    ],
)
def test_corrupt_release_archives_fail_closed_in_api_and_cli(
    tmp_path: Path, capsys, pattern: str, archive_label: str
):
    dist = _write_artifacts(tmp_path, "0.1.0")
    archive = _corrupt_distribution(dist, pattern)
    policy = _write_policy(tmp_path)
    expected = f"Could not inspect {archive_label} {archive.name}"

    with pytest.raises(validator.ReleaseArtifactError, match=expected):
        validator.validate_release_artifacts(
            channel="pypi",
            tag="v0.1.0",
            ref_type="tag",
            dist_dir=dist,
            policy_path=policy,
        )

    result = validator.main(
        [
            "--channel",
            "pypi",
            "--tag",
            "v0.1.0",
            "--ref-type",
            "tag",
            "--dist-dir",
            str(dist),
            "--policy",
            str(policy),
        ]
    )
    output = capsys.readouterr()
    assert result == 1
    assert expected in output.err
    assert "Traceback" not in output.err
