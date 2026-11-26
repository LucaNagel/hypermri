"""Security tests for the real-data download helper."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_download_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "download_test_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hypermri_download_test_data", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_test_data = _load_download_script()


def _materialize_cached_archive(archive_path: Path, data_root: Path) -> int:
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    dataset = {
        "relative_path": "demo-study",
        "archive_url": f"https://example.invalid/{archive_path.name}",
        "checksum": f"sha256:{checksum}",
        "required_paths": ["payload.txt"],
    }
    return download_test_data.materialize_dataset(
        "demo",
        dataset,
        data_root,
        use_cached_archive=True,
        allow_disabled=False,
    )


def _archive_path(data_root: Path, name: str) -> Path:
    path = data_root / ".downloads" / name
    path.parent.mkdir(parents=True)
    return path


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize(
    "url",
    ["http://example.invalid/data.zip", "file:///tmp/data.zip", "data.zip"],
)
def test_download_file_rejects_non_https_urls(tmp_path: Path, url: str):
    with pytest.raises(download_test_data.DatasetError, match="HTTPS"):
        download_test_data.download_file(url, tmp_path / "data.zip")


def test_download_file_removes_partial_file_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class BrokenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int) -> bytes:
            raise OSError("simulated interrupted download")

    monkeypatch.setattr(
        download_test_data,
        "urlopen",
        lambda *_args, **_kwargs: BrokenResponse(),
    )
    destination = tmp_path / "data.zip"

    with pytest.raises(OSError, match="interrupted"):
        download_test_data.download_file(
            "https://example.invalid/data.zip",
            destination,
        )

    assert not destination.exists()
    assert not (tmp_path / "data.zip.part").exists()


def test_download_file_verifies_before_publishing_to_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"incomplete archive"
    monkeypatch.setattr(
        download_test_data,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    destination = tmp_path / "data.zip"

    with pytest.raises(download_test_data.DatasetError, match="size mismatch"):
        download_test_data.download_file(
            "https://example.invalid/data.zip",
            destination,
            expected_size=len(payload) + 1,
        )

    assert not destination.exists()
    assert not (tmp_path / "data.zip.part").exists()


def test_existing_files_without_matching_provenance_are_not_accepted(tmp_path: Path):
    data_root = tmp_path / "data"
    dataset_root = data_root / "demo-study"
    dataset_root.mkdir(parents=True)
    (dataset_root / "payload.txt").write_text("unverified", encoding="utf-8")
    dataset = {
        "relative_path": "demo-study",
        "archive_name": "demo.zip",
        "archive_size_bytes": 123,
        "checksum": "sha256:" + "a" * 64,
        "required_paths": ["payload.txt"],
    }

    assert download_test_data.dataset_files_are_present(data_root, dataset)
    assert not download_test_data.dataset_is_present(data_root, "demo", dataset)

    download_test_data.write_provenance(data_root, "demo", dataset)
    assert download_test_data.dataset_is_present(data_root, "demo", dataset)

    dataset["checksum"] = "sha256:" + "b" * 64
    assert not download_test_data.dataset_is_present(data_root, "demo", dataset)


def test_unreleased_dataset_cannot_be_overridden_or_read_from_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    """Release status blocks even --allow-disabled before cache/network access."""
    data_root = tmp_path / "data"
    cached_dataset = data_root / "demo-study"
    cached_dataset.mkdir(parents=True)
    (cached_dataset / "payload.txt").write_text("must not be used")
    dataset = {
        "enabled": False,
        "release_status": "unreleased",
        "disabled_reason": "No public dataset has been approved.",
        "relative_path": "demo-study",
        "archive_url": "https://example.invalid/demo.zip",
        "required_paths": ["payload.txt"],
    }

    monkeypatch.setattr(
        download_test_data,
        "download_file",
        lambda *_args, **_kwargs: pytest.fail("network access was attempted"),
    )

    result = download_test_data.materialize_dataset(
        "demo",
        dataset,
        data_root,
        use_cached_archive=True,
        allow_disabled=True,
    )

    assert result == 2
    assert "unavailable (unreleased)" in capsys.readouterr().out
    assert cached_dataset.exists()


def test_materialize_dataset_extracts_safe_tar_archive(tmp_path: Path):
    data_root = tmp_path / "data"
    archive_path = _archive_path(data_root, "demo.tar")
    with tarfile.open(archive_path, "w") as archive:
        _add_tar_bytes(archive, "demo-study/payload.txt", b"safe payload")

    result = _materialize_cached_archive(archive_path, data_root)

    assert result == 0
    assert (data_root / "demo-study" / "payload.txt").read_bytes() == b"safe payload"


def test_materialize_dataset_extracts_safe_zip_archive(tmp_path: Path):
    data_root = tmp_path / "data"
    archive_path = _archive_path(data_root, "demo.zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("demo-study/payload.txt", "safe payload")

    result = _materialize_cached_archive(archive_path, data_root)

    assert result == 0
    assert (data_root / "demo-study" / "payload.txt").read_text() == "safe payload"
    assert download_test_data.provenance_is_valid(
        data_root,
        "demo",
        {
            "relative_path": "demo-study",
            "archive_url": f"https://example.invalid/{archive_path.name}",
            "checksum": f"sha256:{hashlib.sha256(archive_path.read_bytes()).hexdigest()}",
            "required_paths": ["payload.txt"],
        },
    )


def test_materialize_dataset_rejects_wrong_archive_size(
    tmp_path: Path,
    capsys,
):
    data_root = tmp_path / "data"
    archive_path = _archive_path(data_root, "demo.zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("demo-study/payload.txt", "must not be extracted")
    dataset = {
        "relative_path": "demo-study",
        "archive_url": f"https://example.invalid/{archive_path.name}",
        "archive_size_bytes": archive_path.stat().st_size + 1,
        "required_paths": ["payload.txt"],
    }

    result = download_test_data.materialize_dataset(
        "demo",
        dataset,
        data_root,
        use_cached_archive=True,
        allow_disabled=False,
    )

    assert result == 1
    assert "size mismatch" in capsys.readouterr().out
    assert not (data_root / "demo-study" / "payload.txt").exists()


@pytest.mark.parametrize("member_name", ["../escaped.txt", "/absolute.txt"])
def test_materialize_dataset_rejects_tar_paths_outside_target(
    tmp_path: Path,
    member_name: str,
):
    data_root = tmp_path / "data"
    archive_path = _archive_path(data_root, "demo.tar")
    with tarfile.open(archive_path, "w") as archive:
        _add_tar_bytes(archive, "demo-study/payload.txt", b"must not be extracted")
        _add_tar_bytes(archive, member_name, b"escape")

    result = _materialize_cached_archive(archive_path, data_root)

    assert result == 1
    assert not (tmp_path / "escaped.txt").exists()
    assert not (data_root / "demo-study" / "payload.txt").exists()


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE],
    ids=["symlink", "hardlink", "character-device", "fifo"],
)
def test_materialize_dataset_rejects_tar_links_and_special_files(
    tmp_path: Path,
    member_type: bytes,
):
    data_root = tmp_path / "data"
    archive_path = _archive_path(data_root, "demo.tar")
    with tarfile.open(archive_path, "w") as archive:
        _add_tar_bytes(archive, "demo-study/payload.txt", b"must not be extracted")
        member = tarfile.TarInfo("demo-study/unsafe-entry")
        member.type = member_type
        member.linkname = "../../escaped.txt"
        archive.addfile(member)

    result = _materialize_cached_archive(archive_path, data_root)

    assert result == 1
    assert not (tmp_path / "escaped.txt").exists()
    assert not (data_root / "demo-study" / "payload.txt").exists()


def test_shared_reference_registry_retains_pinned_download_identity():
    """The test downloader consumes the same entry as the package registry."""
    from hypermri.datasets import load_registry

    downloader = _load_download_script()
    actual = downloader.load_manifest(downloader.DEFAULT_MANIFEST)
    registry = load_registry()
    assert actual["datasets"] == registry["reference_datasets"]
    assert actual["data_root_env"] == registry["test_data_root_env"]
