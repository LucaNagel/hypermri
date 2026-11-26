#!/usr/bin/env python3
"""Download and verify optional real-world test datasets.

The repository tracks only a small manifest. The actual MRI data should live in
an external archive, local cache, or CI secret-backed storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from hypermri.datasets import DatasetError, extract_archive_safely

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "src" / "hypermri" / "datasets" / "registry.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "tests" / "data"
USER_AGENT = "hypermri-realdata-tests (+https://github.com/LucaNagel/hypermri)"
PROVENANCE_FILENAME = ".hypermri-test-data.json"


def load_manifest(path: Path) -> dict:
    """Read reference data from the shared registry or a legacy custom manifest.

    Parameters
    ----------
    path : pathlib.Path
        Shared dataset registry or existing user-supplied test manifest.

    Returns
    -------
    dict
        Test-only datasets and their cache environment setting. This reads JSON
        eagerly and performs no downloads. Public tutorial placeholders remain
        outside test selection. Existing custom manifest layouts are preserved.

    Raises
    ------
    OSError
        If the file cannot be read.
    json.JSONDecodeError
        If the JSON is malformed.
    """
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if "reference_datasets" in manifest:
        from hypermri.datasets.fetch import _validate_registry

        manifest = _validate_registry(manifest, path)
        return {
            "version": manifest["version"],
            "data_root_env": manifest["test_data_root_env"],
            "datasets": manifest["reference_datasets"],
        }
    return manifest


def resolve_data_root(manifest: dict, override: str | None) -> Path:
    """Resolve the local cache directory for real test data."""
    env_name = manifest.get("data_root_env", "HYPERMRI_TEST_DATA")
    root = override or os.environ.get(env_name)
    if root:
        return Path(root).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def dataset_path(data_root: Path, dataset: dict) -> Path:
    """Return the expected local dataset directory."""
    return data_root / dataset["relative_path"]


def is_lfs_pointer(path: Path) -> bool:
    """Return True when a file is an unresolved Git LFS pointer."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            prefix = stream.read(128)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def dataset_files_are_present(data_root: Path, dataset: dict) -> bool:
    """Check whether a dataset and its required files exist locally."""
    root = dataset_path(data_root, dataset)
    if not root.exists():
        return False
    for rel_path in dataset.get("required_paths", []):
        required_path = root / rel_path
        if not required_path.exists() or is_lfs_pointer(required_path):
            return False
    return True


def expected_provenance(dataset_name: str, dataset: dict) -> dict:
    """Return the manifest identity written after verified extraction."""
    checksum = dataset.get("checksum", "").strip().lower()
    if not checksum and dataset.get("sha256", "").strip():
        checksum = f"sha256:{dataset['sha256'].strip().lower()}"
    return {
        "schema_version": 1,
        "dataset_id": dataset_name,
        "relative_path": dataset["relative_path"],
        "archive_name": dataset.get("archive_name", ""),
        "archive_size_bytes": dataset.get("archive_size_bytes"),
        "checksum": checksum,
    }


def provenance_is_valid(data_root: Path, dataset_name: str, dataset: dict) -> bool:
    """Check that local files came from the currently pinned manifest entry."""
    marker = dataset_path(data_root, dataset) / PROVENANCE_FILENAME
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == expected_provenance(dataset_name, dataset)


def write_provenance(data_root: Path, dataset_name: str, dataset: dict) -> None:
    """Atomically record the verified archive identity beside extracted data."""
    marker = dataset_path(data_root, dataset) / PROVENANCE_FILENAME
    partial = marker.with_name(f"{marker.name}.part")
    try:
        partial.write_text(
            json.dumps(
                expected_provenance(dataset_name, dataset),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        partial.replace(marker)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def dataset_is_present(data_root: Path, dataset_name: str, dataset: dict) -> bool:
    """Check required files and their verified-archive provenance marker."""
    return dataset_files_are_present(data_root, dataset) and provenance_is_valid(
        data_root,
        dataset_name,
        dataset,
    )


def parse_checksum(checksum: str) -> tuple[str, str]:
    """Return checksum algorithm and expected digest."""
    checksum = checksum.strip().lower()
    if not checksum:
        return "", ""
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "sha256", checksum
    return algorithm.strip(), expected.strip()


def file_checksum(path: Path, algorithm: str) -> str:
    """Compute a digest for a file."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_name_from_url(url: str, dataset_name: str) -> str:
    """Derive a stable archive filename from a URL."""
    parsed_name = Path(urlparse(url).path).name
    return parsed_name or f"{dataset_name}.archive"


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    checksum: str = "",
) -> None:
    """Download and verify an HTTPS URL before publishing it to the cache."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise DatasetError("Real test-data downloads require an HTTPS URL.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=120) as response:
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if expected_size is not None:
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise DatasetError(
                    "Downloaded archive size mismatch: "
                    f"expected {expected_size} bytes, got {actual_size} bytes."
                )
        algorithm, expected = parse_checksum(checksum)
        if algorithm:
            actual = file_checksum(partial, algorithm)
            if actual.lower() != expected:
                raise DatasetError(
                    "Downloaded archive checksum mismatch: "
                    f"expected {algorithm}:{expected}, got {algorithm}:{actual}."
                )
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def get_archive_url(dataset: dict) -> str:
    """Resolve the archive URL from an environment variable or manifest field."""
    env_name = dataset.get("archive_url_env")
    if env_name:
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url
    return dataset.get("archive_url", "").strip()


def materialize_dataset(
    dataset_name: str,
    dataset: dict,
    data_root: Path,
    *,
    use_cached_archive: bool,
    allow_disabled: bool,
) -> int:
    """Ensure one dataset exists locally."""
    release_status = dataset.get("release_status", "released")
    if release_status != "released":
        reason = dataset.get(
            "disabled_reason",
            "The dataset has not been approved for distribution.",
        )
        print(
            f"{dataset_name}: unavailable ({release_status}). {reason} "
            "No local cache or network access was attempted."
        )
        return 2

    expected_path = dataset_path(data_root, dataset)
    if dataset.get("enabled") is False and not allow_disabled:
        print(
            f"{dataset_name}: disabled in src/hypermri/datasets/registry.json. "
            "Fill the archive metadata and pass --allow-disabled to test it before activation."
        )
        return 2

    if dataset_is_present(data_root, dataset_name, dataset):
        print(f"{dataset_name}: already present at {expected_path}")
        return 0

    url = get_archive_url(dataset)
    if not url:
        if dataset.get("storage") == "git-lfs":
            lfs_include = dataset.get("lfs_include", dataset["relative_path"])
            print(
                f"{dataset_name}: missing at {expected_path}\n"
                "This dataset is expected to be stored with Git LFS. Run:\n"
                f"  git lfs pull --include {lfs_include!r} --exclude ''"
            )
            return 2
        env_name = dataset.get(
            "archive_url_env", "the dataset URL environment variable"
        )
        print(
            f"{dataset_name}: missing at {expected_path}\n"
            f"Set {env_name} or fill archive_url in src/hypermri/datasets/registry.json."
        )
        return 2

    downloads_dir = data_root / ".downloads"
    archive_path = downloads_dir / archive_name_from_url(url, dataset_name)
    expected_size_value = dataset.get("archive_size_bytes")
    expected_size = (
        int(expected_size_value) if expected_size_value is not None else None
    )
    checksum = dataset.get("checksum", "").strip()
    if not checksum and dataset.get("sha256", "").strip():
        checksum = f"sha256:{dataset['sha256']}"
    if archive_path.exists() and use_cached_archive:
        print(f"{dataset_name}: reusing cached archive {archive_path}")
    else:
        print(f"{dataset_name}: downloading {url}")
        try:
            download_file(
                url,
                archive_path,
                expected_size=expected_size,
                checksum=checksum,
            )
        except (DatasetError, OSError) as exc:
            print(f"{dataset_name}: download failed: {exc}")
            return 1

    if expected_size is not None:
        actual_size = archive_path.stat().st_size
        if actual_size != expected_size:
            print(
                f"{dataset_name}: size mismatch for {archive_path}\n"
                f"expected: {expected_size} bytes\n"
                f"actual:   {actual_size} bytes"
            )
            return 1

    algorithm, expected = parse_checksum(checksum)
    if algorithm:
        actual = file_checksum(archive_path, algorithm)
        if actual.lower() != expected:
            print(
                f"{dataset_name}: checksum mismatch for {archive_path}\n"
                f"expected: {algorithm}:{expected}\n"
                f"actual:   {algorithm}:{actual}"
            )
            return 1

    try:
        extract_archive_safely(archive_path, data_root)
    except DatasetError as exc:
        print(f"{dataset_name}: refusing to extract unsafe archive: {exc}")
        return 1

    if not dataset_files_are_present(data_root, dataset):
        print(
            f"{dataset_name}: archive extracted, but expected path is still missing: "
            f"{expected_path}"
        )
        return 1

    try:
        write_provenance(data_root, dataset_name, dataset)
    except OSError as exc:
        print(f"{dataset_name}: could not record verified provenance: {exc}")
        return 1

    print(f"{dataset_name}: ready at {expected_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the real-data manifest JSON file.",
    )
    parser.add_argument(
        "--data-root",
        help="Local data cache root. Defaults to HYPERMRI_TEST_DATA or tests/data.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key to download, or 'all'.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Download the archive even if a cached copy exists.",
    )
    parser.add_argument(
        "--allow-disabled",
        action="store_true",
        help=(
            "Allow testing entries marked enabled=false. Entries whose "
            "release_status is not 'released' remain blocked."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Download requested datasets."""
    args = parse_args()
    manifest = load_manifest(args.manifest)
    datasets = manifest.get("datasets", {})
    data_root = resolve_data_root(manifest, args.data_root)

    if args.dataset == "all":
        selected = datasets.items()
    else:
        if args.dataset not in datasets:
            print(f"Unknown dataset {args.dataset!r}. Available: {sorted(datasets)}")
            return 1
        selected = [(args.dataset, datasets[args.dataset])]

    exit_code = 0
    for dataset_name, dataset in selected:
        result = materialize_dataset(
            dataset_name,
            dataset,
            data_root,
            use_cached_archive=not args.no_cache,
            allow_disabled=args.allow_disabled,
        )
        exit_code = max(exit_code, result)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
