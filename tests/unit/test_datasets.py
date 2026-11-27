"""Tests for public example dataset helpers."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from hypermri.datasets import fetch_example_data, get_dataset_info, list_datasets
from hypermri.datasets.fetch import (
    DatasetError,
    _safe_extract_tar,
    _safe_extract_zip,
    load_registry,
)


def test_packaged_registry_releases_no_example_yet():
    """No example is released; the placeholder stays blocked."""
    assert list_datasets() == []
    assert list_datasets(include_unavailable=True) == ["tutorial"]

    info = get_dataset_info("tutorial")
    assert info["enabled"] is False
    assert info["release_status"] == "unreleased"
    assert "archive_url" not in info
    assert "record_id" not in info


def test_packaged_registry_refuses_fetch_before_filesystem_or_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An unreleased dataset is blocked before creating a cache or resolving files."""
    cache = tmp_path / "cache"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("remote file resolution must not be attempted")

    monkeypatch.setattr(
        "hypermri.datasets.fetch._resolve_dataset_files",
        fail_if_called,
    )

    with pytest.raises(
        DatasetError,
        match="not publicly available.*No cache or network access was attempted",
    ):
        fetch_example_data("tutorial", data_home=cache)

    assert not cache.exists()


def test_dataset_cli_list_says_when_nothing_is_released(capsys):
    """The CLI list states that no example is released instead of printing nothing."""
    from hypermri.datasets.fetch import main

    assert main(["--list"]) == 0
    assert capsys.readouterr().out.strip() == (
        "No public example datasets are currently released."
    )


def test_fetch_example_data_extracts_archive_from_registry(tmp_path: Path):
    """A registry archive entry downloads, verifies, and extracts into the cache."""
    archive_path = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("demo-study/ScanProgram.scanProgram", "scan program")
        archive.writestr("demo-study/subject", "subject")

    checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_root_env": "HYPERMRI_DATA",
                "datasets": {
                    "demo": {
                        "description": "Local demo archive",
                        "source": "archive",
                        "archive_url": archive_path.as_uri(),
                        "checksum": f"md5:{checksum}",
                        "cache_subdir": "demo",
                        "relative_path": "demo-study",
                        "required_paths": [
                            "ScanProgram.scanProgram",
                            "subject",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    data_path = fetch_example_data(
        "demo",
        data_home=tmp_path / "cache",
        registry_path=registry_path,
        allow_local_files=True,
        quiet=True,
    )

    assert data_path == tmp_path / "cache" / "demo" / "demo-study"
    assert (data_path / "ScanProgram.scanProgram").read_text() == "scan program"
    assert (data_path / "subject").read_text() == "subject"
    assert (data_path.parent / ".hypermri_dataset.json").exists()


def _write_archive_registry(tmp_path: Path, archive_url: str) -> Path:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_root_env": "HYPERMRI_DATA",
                "datasets": {
                    "demo": {
                        "source": "archive",
                        "archive_url": archive_url,
                        "cache_subdir": "demo",
                        "required_paths": ["payload.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path


def test_load_registry_rejects_missing_datasets_section(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"version": 1, "data_root_env": "HYPERMRI_DATA"}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="datasets"):
        load_registry(registry_path)


def test_load_registry_rejects_an_available_dataset_without_its_source_field(
    tmp_path: Path,
):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_root_env": "HYPERMRI_DATA",
                "datasets": {"demo": {"description": "missing source"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="demo.*source"):
        load_registry(registry_path)


def test_fetch_example_data_rejects_local_urls_by_default(tmp_path: Path):
    """Local file access is available only through the explicit test-mode flag."""
    archive_path = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("payload.txt", "payload")
    registry_path = _write_archive_registry(tmp_path, archive_path.as_uri())

    with pytest.raises(DatasetError, match="HTTPS"):
        fetch_example_data(
            "demo",
            data_home=tmp_path / "cache",
            registry_path=registry_path,
            quiet=True,
        )


def test_fetch_example_data_rejects_http_urls(tmp_path: Path):
    """Registry downloads cannot silently downgrade transport security."""
    registry_path = _write_archive_registry(tmp_path, "http://example.invalid/demo.zip")

    with pytest.raises(DatasetError, match="HTTPS"):
        fetch_example_data(
            "demo",
            data_home=tmp_path / "cache",
            registry_path=registry_path,
            quiet=True,
        )


@pytest.mark.parametrize(
    "member_name", ["../escaped.txt", "/absolute.txt", "C:\\escaped.txt"]
)
def test_safe_extract_zip_rejects_unsafe_paths(tmp_path: Path, member_name: str):
    """Zip traversal and absolute paths are rejected before any member is written."""
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", "must not be extracted")
        archive.writestr(member_name, "escape")
    target = tmp_path / "target"

    with pytest.raises(DatasetError, match="Unsafe"):
        _safe_extract_zip(archive_path, target)

    assert not (tmp_path / "escaped.txt").exists()
    assert not (target / "safe.txt").exists()


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO])
def test_safe_extract_zip_rejects_special_entries(tmp_path: Path, file_type: int):
    """Unix-style links and special files embedded in zip metadata are rejected."""
    archive_path = tmp_path / "special.zip"
    member = zipfile.ZipInfo("special-entry")
    member.create_system = 3
    member.external_attr = (file_type | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "../../escaped.txt")

    with pytest.raises(DatasetError, match="Unsafe"):
        _safe_extract_zip(archive_path, tmp_path / "target")

    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE],
)
def test_safe_extract_tar_rejects_links_and_special_entries(
    tmp_path: Path, member_type: bytes
):
    """Tar links, devices, and other special entries cannot escape extraction."""
    archive_path = tmp_path / "special.tar"
    member = tarfile.TarInfo("special-entry")
    member.type = member_type
    member.linkname = "../../escaped.txt"
    with tarfile.open(archive_path, "w") as archive:
        archive.addfile(member)

    with pytest.raises(DatasetError, match="Unsafe"):
        _safe_extract_tar(archive_path, tmp_path / "target")

    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_tar_rejects_traversal_before_writing(tmp_path: Path):
    """All tar paths are validated before extraction begins."""
    archive_path = tmp_path / "traversal.tar"
    safe_data = b"must not be extracted"
    escaped_data = b"escape"
    with tarfile.open(archive_path, "w") as archive:
        safe_member = tarfile.TarInfo("safe.txt")
        safe_member.size = len(safe_data)
        archive.addfile(safe_member, io.BytesIO(safe_data))
        escaped_member = tarfile.TarInfo("../escaped.txt")
        escaped_member.size = len(escaped_data)
        archive.addfile(escaped_member, io.BytesIO(escaped_data))
    target = tmp_path / "target"

    with pytest.raises(DatasetError, match="Unsafe"):
        _safe_extract_tar(archive_path, target)

    assert not (tmp_path / "escaped.txt").exists()
    assert not (target / "safe.txt").exists()


def test_safe_extract_tar_extracts_regular_files_and_directories(tmp_path: Path):
    """Hardening preserves ordinary tar archive extraction."""
    archive_path = tmp_path / "safe.tar"
    payload = b"payload"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("dataset")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo("dataset/payload.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    target = tmp_path / "target"

    _safe_extract_tar(archive_path, target)

    assert (target / "dataset" / "payload.txt").read_bytes() == payload


def test_reference_corpus_has_one_owner_apart_from_the_examples():
    """The shared registry separates pinned integration references from examples."""
    registry = load_registry()
    assert registry["test_data_root_env"] == "HYPERMRI_TEST_DATA"
    reference = registry["reference_datasets"]["brukerapi_pv601_lego_phantom"]
    assert reference["enabled"] is True
    assert reference["release_status"] == "released"
    assert reference["source"] == "archive"
    assert reference["checksum"].startswith("sha256:")
    assert reference["stable_raw_cases"]["csi"] == 30
    assert reference["archive_size_bytes"] == 783667029
    assert "brukerapi_pv601_lego_phantom" not in list_datasets()
    assert not (
        Path(__file__).resolve().parents[1] / "real_data_manifest.json"
    ).exists()


@pytest.mark.parametrize(
    "field, value",
    [
        ("reference_datasets", []),
        ("test_data_root_env", None),
    ],
)
def test_reference_registry_rejects_invalid_scope_fields(tmp_path, field, value):
    registry = load_registry()
    registry[field] = value
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(DatasetError, match=field):
        load_registry(path)


def test_dataset_registry_rejects_duplicate_json_keys(tmp_path):
    """Prevent a second dataset field from silently replacing pinned metadata."""
    path = tmp_path / "registry.json"
    path.write_text(
        '{"version": 1, "version": 2, "data_root_env": "CACHE", "datasets": {}}'
    )
    with pytest.raises(DatasetError, match="duplicate key 'version'"):
        load_registry(path)


def _write_token_registry(tmp_path: Path, archive_path: Path) -> Path:
    """Write a registry whose archive accepts a token from TEST_DATASET_TOKEN."""
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_root_env": "HYPERMRI_DATA",
                "datasets": {
                    "demo": {
                        "source": "archive",
                        "archive_url": "https://zenodo.example/files/demo.zip?download=1",
                        "archive_name": "demo.zip",
                        "checksum": f"sha256:{checksum}",
                        "source_url": "https://zenodo.example/records/1",
                        "access_token_env": "TEST_DATASET_TOKEN",
                        "required_paths": ["payload.txt"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path


def _demo_zip(tmp_path: Path) -> Path:
    archive_path = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("payload.txt", "payload")
    return archive_path


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, url: str = "https://zenodo.example/files/x"):
        super().__init__(data)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@pytest.mark.parametrize("token", [None, "s3cret-token"])
def test_fetch_example_data_appends_the_token_only_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token
):
    """The token is added to the download URL and never written to the cache."""
    archive_path = _demo_zip(tmp_path)
    registry_path = _write_token_registry(tmp_path, archive_path)
    if token is None:
        monkeypatch.delenv("TEST_DATASET_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TEST_DATASET_TOKEN", token)
    requested = []

    def fake_urlopen(request, timeout):
        requested.append(request.full_url)
        return _FakeResponse(archive_path.read_bytes())

    monkeypatch.setattr("hypermri.datasets.fetch.urlopen", fake_urlopen)
    path = fetch_example_data(
        "demo", data_home=tmp_path / "cache", registry_path=registry_path, quiet=True
    )

    expected = "https://zenodo.example/files/demo.zip?download=1"
    if token is not None:
        expected += f"&token={token}"
    assert requested == [expected]
    assert (path / "payload.txt").read_text() == "payload"
    marker = (path / ".hypermri_dataset.json").read_text()
    assert "s3cret" not in marker


@pytest.mark.parametrize(
    ("token", "hint"),
    [
        (None, "set the environment variable TEST_DATASET_TOKEN"),
        ("bad", "not accepted"),
    ],
)
def test_refused_download_names_the_token_variable_without_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token, hint
):
    """A 403 from a restricted record explains the token and hides the URL."""
    from urllib.error import HTTPError

    registry_path = _write_token_registry(tmp_path, _demo_zip(tmp_path))
    if token is None:
        monkeypatch.delenv("TEST_DATASET_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TEST_DATASET_TOKEN", token)

    def refuse(request, timeout):
        raise HTTPError(request.full_url, 403, "FORBIDDEN", {}, None)

    monkeypatch.setattr("hypermri.datasets.fetch.urlopen", refuse)
    with pytest.raises(DatasetError, match="HTTP 403") as excinfo:
        fetch_example_data(
            "demo",
            data_home=tmp_path / "cache",
            registry_path=registry_path,
            quiet=True,
        )

    message = str(excinfo.value)
    assert hint in message
    assert "token=" not in message
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"source": "archive", "archive_url": "https://x/a.zip"}, "upper case"),
        ({"source": "zenodo", "record_id": "1"}, "only with source 'archive'"),
    ],
)
def test_registry_rejects_invalid_access_token_env(tmp_path: Path, entry, match):
    """The token variable must be an upper-case name on an archive dataset."""
    env_name = "lower-case" if entry["source"] == "archive" else "TEST_DATASET_TOKEN"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_root_env": "HYPERMRI_DATA",
                "datasets": {"demo": {**entry, "access_token_env": env_name}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match=match):
        load_registry(registry_path)


def test_is_dataset_cached_checks_required_files_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cache check needs only the required files and no network."""
    from hypermri.datasets import is_dataset_cached

    registry_path = _write_token_registry(tmp_path, _demo_zip(tmp_path))
    monkeypatch.setattr(
        "hypermri.datasets.fetch.urlopen",
        lambda *_args, **_kwargs: pytest.fail("no network access expected"),
    )
    cache = tmp_path / "cache"
    assert not is_dataset_cached("demo", data_home=cache, registry_path=registry_path)
    (cache / "demo").mkdir(parents=True)
    (cache / "demo" / "payload.txt").write_text("payload")
    assert is_dataset_cached("demo", data_home=cache, registry_path=registry_path)


def test_redirect_to_the_zenodo_login_counts_as_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Zenodo redirects restricted files to /login; that is no data download."""
    registry_path = _write_token_registry(tmp_path, _demo_zip(tmp_path))
    monkeypatch.delenv("TEST_DATASET_TOKEN", raising=False)
    monkeypatch.setattr(
        "hypermri.datasets.fetch.urlopen",
        lambda request, timeout: _FakeResponse(
            b"<html>login</html>", "https://zenodo.example/login/?next=%2Ffiles"
        ),
    )

    with pytest.raises(DatasetError, match="HTTP 401.*TEST_DATASET_TOKEN"):
        fetch_example_data(
            "demo",
            data_home=tmp_path / "cache",
            registry_path=registry_path,
            quiet=True,
        )
    assert not (tmp_path / "cache" / "demo" / ".downloads" / "demo.zip").exists()
