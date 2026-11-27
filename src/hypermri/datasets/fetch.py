"""Resolve, download, and cache approved public example datasets.

The package ships only a small registry and the reusable download machinery.
Real MRI data stays outside the wheel and is fetched into a local cache only
after a registry entry has explicitly been approved for public release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.error import HTTPError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from hypermri.config.mappings import check_fields
from hypermri.exceptions import ConfigurationError

REGISTRY_RESOURCE = "registry.json"
ZENODO_API_URL = "https://zenodo.org/api/records/{record_id}"
USER_AGENT = "hypermri-datasets (+https://github.com/LucaNagel/hypermri)"
_ENV_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*")


class DatasetError(RuntimeError):
    """Raised when an example dataset cannot be resolved or verified."""


@dataclass(frozen=True)
class DatasetFile:
    """File entry resolved from a static archive or a Zenodo record."""

    key: str
    url: str
    checksum: str = ""
    size: int | None = None


def _unique_registry_object(pairs):
    """Reject duplicate JSON fields before they can replace dataset metadata."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def load_registry(registry_path: str | Path | None = None) -> dict:
    """Load and validate the example dataset registry.

    Parameters
    ----------
    registry_path : str or pathlib.Path, optional
        JSON registry to read. The packaged ``registry.json`` is used when
        omitted.

    Returns
    -------
    dict
        Registry with the required top-level fields ``version``,
        ``data_root_env`` and ``datasets``. Optional ``reference_datasets``
        retains the pinned test corpus separately from approved public examples;
        ``test_data_root_env`` names its opt-in local cache override. ``enabled`` and
        ``release_status`` remain optional dataset fields and retain their
        documented defaults of ``True`` and ``"released"``.

    Raises
    ------
    DatasetError
        If the JSON file cannot be read or its schema is invalid. The error
        identifies the registry path and, for dataset entries, the dataset
        name.
    """
    if registry_path is not None:
        path = Path(registry_path).expanduser()
        try:
            registry = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_registry_object,
            )
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Cannot read dataset registry {path}: {exc}") from exc
        return _validate_registry(registry, path)

    resource = resource_files("hypermri.datasets").joinpath(REGISTRY_RESOURCE)
    path = Path(str(resource))
    try:
        registry = json.loads(
            resource.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_registry_object,
        )
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read dataset registry {path}: {exc}") from exc
    return _validate_registry(registry, path)


_DATASET_FIELDS = (
    "enabled",
    "release_status",
    "description",
    "disabled_reason",
    "source",
    "archive_url",
    "archive_name",
    "record_id",
    "doi",
    "checksum",
    "cache_subdir",
    "relative_path",
    "required_paths",
    "creator",
    "title",
    "dataset_version",
    "publisher",
    "source_url",
    "license",
    "license_url",
    "archive_size_bytes",
    "stable_raw_cases",
    "access_token_env",
)


def _validate_registry(registry: object, path: Path) -> dict:
    """Validate the JSON dataset registry schema and return its mapping."""
    try:
        validated = check_fields(
            registry,
            required=("version", "data_root_env", "datasets"),
            optional=("test_data_root_env", "reference_datasets"),
            where="the top level",
            config_type="dataset",
            path=path,
        )
    except ConfigurationError as exc:
        raise DatasetError(str(exc)) from exc

    if not isinstance(validated["version"], int) or isinstance(
        validated["version"], bool
    ):
        raise DatasetError(
            f"Dataset registry {path} field 'version' must be an integer."
        )
    if (
        not isinstance(validated["data_root_env"], str)
        or not validated["data_root_env"].strip()
    ):
        raise DatasetError(
            f"Dataset registry {path} field 'data_root_env' must be a non-empty string."
        )
    if not isinstance(validated["datasets"], dict):
        raise DatasetError(
            f"Dataset registry {path} field 'datasets' must be a mapping."
        )

    references = validated.get("reference_datasets", {})
    if not isinstance(references, dict):
        raise DatasetError(
            f"Dataset registry {path} reference_datasets must be a mapping."
        )
    if references and (
        not isinstance(validated.get("test_data_root_env"), str)
        or not validated["test_data_root_env"].strip()
    ):
        raise DatasetError(
            f"Dataset registry {path} requires test_data_root_env for references."
        )
    if set(references) & set(validated["datasets"]):
        raise DatasetError(f"Dataset registry {path} repeats a dataset across scopes.")
    entries = {**validated["datasets"], **references}
    for name, dataset in entries.items():
        if not isinstance(name, str) or not name:
            raise DatasetError(
                f"Dataset registry {path} has an empty or non-string name."
            )
        try:
            entry = check_fields(
                dataset,
                required=(),
                optional=_DATASET_FIELDS,
                where=f"dataset {name!r}",
                config_type="dataset",
                path=path,
            )
        except ConfigurationError as exc:
            raise DatasetError(str(exc)) from exc
        _validate_dataset_entry(name, entry, path)
    return dict(validated)


def _validate_dataset_entry(name: str, dataset: dict, path: Path) -> None:
    """Validate fields whose requirements depend on a dataset source."""
    if "enabled" in dataset and not isinstance(dataset["enabled"], bool):
        raise DatasetError(
            f"Dataset {name!r} in {path} field 'enabled' must be a boolean."
        )
    for field in ("release_status", "description", "disabled_reason"):
        if field in dataset and not isinstance(dataset[field], str):
            raise DatasetError(
                f"Dataset {name!r} in {path} field {field!r} must be a string."
            )
    if "required_paths" in dataset and (
        not isinstance(dataset["required_paths"], list)
        or not all(isinstance(value, str) for value in dataset["required_paths"])
    ):
        raise DatasetError(
            f"Dataset {name!r} in {path} field 'required_paths' must be a list of strings."
        )
    for field in (
        "creator",
        "title",
        "dataset_version",
        "publisher",
        "source_url",
        "license",
        "license_url",
    ):
        if field in dataset and not isinstance(dataset[field], str):
            raise DatasetError(
                f"Dataset {name!r} in {path} field {field!r} must be a string."
            )
    if "archive_size_bytes" in dataset:
        size = dataset["archive_size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise DatasetError(
                f"Dataset {name!r} in {path} archive_size_bytes must be positive."
            )
    if "stable_raw_cases" in dataset:
        cases = dataset["stable_raw_cases"]
        if not isinstance(cases, dict) or not all(
            isinstance(key, str)
            and key
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for key, value in cases.items()
        ):
            raise DatasetError(
                f"Dataset {name!r} in {path} stable_raw_cases must map names to positive scan indices."
            )
    if "access_token_env" in dataset:
        env_name = dataset["access_token_env"]
        if not isinstance(env_name, str) or not _ENV_NAME_PATTERN.fullmatch(env_name):
            raise DatasetError(
                f"Dataset {name!r} in {path} field 'access_token_env' must name an "
                "environment variable in upper case, for example "
                "'HYPERMRI_ZENODO_TOKEN'."
            )
        if dataset.get("source") != "archive":
            raise DatasetError(
                f"Dataset {name!r} in {path} sets 'access_token_env', which is "
                "supported only with source 'archive'."
            )
    available = (
        dataset.get("enabled", True) is not False
        and dataset.get("release_status", "released") == "released"
    )
    if not available:
        return
    source = dataset.get("source")
    if source not in {"archive", "zenodo"}:
        raise DatasetError(
            f"Dataset {name!r} in {path} must set 'source' to 'archive' or 'zenodo'."
        )
    required_field = "archive_url" if source == "archive" else "record_id"
    if (
        not isinstance(dataset.get(required_field), str)
        or not dataset[required_field].strip()
    ):
        raise DatasetError(
            f"Dataset {name!r} in {path} with source {source!r} requires a non-empty "
            f"{required_field!r}."
        )


def list_datasets(
    registry: dict | None = None,
    *,
    include_unavailable: bool = False,
) -> list[str]:
    """Return example dataset names that are approved for use.

    Parameters
    ----------
    registry : dict, optional
        Parsed dataset registry. The packaged registry is used by default.
    include_unavailable : bool, default=False
        Include disabled placeholders. This is intended for registry
        maintenance and diagnostics, not for selecting downloadable data.

    Returns
    -------
    list of str
        Dataset names in alphabetical order, each accepted by
        :func:`fetch_example_data`.
    """
    if registry is None:
        registry = load_registry()
    datasets = registry.get("datasets", {})
    if include_unavailable:
        return sorted(datasets)
    return sorted(
        name
        for name, dataset in datasets.items()
        if dataset.get("enabled", True) is not False
        and dataset.get("release_status", "released") == "released"
    )


def get_dataset_info(name: str = "tutorial", registry: dict | None = None) -> dict:
    """Return one dataset registry entry.

    Parameters
    ----------
    name : str
        Registry name of the dataset.
    registry : dict, optional
        Parsed dataset registry. The packaged registry is used by default.

    Returns
    -------
    dict
        The registry entry: description, source, DOI, licence, download URL,
        checksum and required files. The returned mapping is the registry's
        own; do not modify it.

    Raises
    ------
    DatasetError
        If no dataset of this name exists. The message lists the released
        names.
    """
    if registry is None:
        registry = load_registry()
    datasets = registry.get("datasets", {})
    try:
        return datasets[name]
    except KeyError as exc:
        available = ", ".join(list_datasets(registry)) or "none"
        raise DatasetError(
            f"Unknown dataset {name!r}. Available datasets: {available}."
        ) from exc


def _require_dataset_available(name: str, dataset: dict) -> None:
    """Reject dataset placeholders before cache or network access."""
    release_status = dataset.get("release_status", "released")
    enabled = dataset.get("enabled", True)
    if enabled is not False and release_status == "released":
        return

    reason = dataset.get("disabled_reason", "").strip()
    if not reason:
        reason = "This dataset has not been approved for public release."
    raise DatasetError(
        f"Dataset {name!r} is not publicly available. {reason} "
        "No cache or network access was attempted."
    )


def get_data_home(
    data_home: str | Path | None = None, registry: dict | None = None
) -> Path:
    """Return the local dataset cache root."""
    if data_home is not None:
        return Path(data_home).expanduser().resolve()

    registry = registry or load_registry()
    env_name = registry.get("data_root_env", "HYPERMRI_DATA")
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "hypermri" / "datasets").resolve()


def dataset_path(data_home: Path, dataset_name: str, dataset: dict) -> Path:
    """Return the expected usable dataset path after download/extraction."""
    cache_subdir = dataset.get("cache_subdir", dataset_name)
    root = data_home / cache_subdir if cache_subdir else data_home
    relative_path = dataset.get("relative_path", "").strip()
    return root / relative_path if relative_path else root


def is_dataset_cached(
    name: str,
    *,
    data_home: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> bool:
    """Return whether a dataset is complete in the local cache.

    The check only looks for the required files of the registry entry in the
    cache; it never downloads, verifies checksums or touches the network.
    :func:`fetch_example_data` returns such a dataset without network access.

    Parameters
    ----------
    name : str
        Registry name of the dataset, for example ``"hp_pyruvate_csi_mouse"``.
    data_home : str or pathlib.Path, optional
        Cache root. Defaults to the environment variable ``HYPERMRI_DATA`` or,
        without it, to the user cache folder of the operating system.
    registry_path : str or pathlib.Path, optional
        Registry JSON file to use instead of the packaged one.

    Returns
    -------
    bool
        ``True`` if every required file of the dataset exists in the cache.

    Raises
    ------
    DatasetError
        If the registry cannot be read or names no dataset ``name``.
    """
    registry = load_registry(registry_path)
    dataset = get_dataset_info(name, registry)
    root = get_data_home(data_home, registry)
    return _dataset_is_present(dataset_path(root, name, dataset), dataset)


def _read_access_token(dataset: dict) -> str | None:
    """Return the access token named by the registry entry, if one is set."""
    env_name = dataset.get("access_token_env")
    if not env_name:
        return None
    return os.environ.get(env_name, "").strip() or None


def _with_access_token(url: str, token: str | None) -> str:
    """Return ``url`` with the Zenodo share-link token as query parameter."""
    if token is None:
        return url
    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key != "token"
    ]
    query.append(("token", token))
    return urlunsplit(parts._replace(query=urlencode(query)))


def _refused_download_message(
    name: str, dataset: dict, file_entry: DatasetFile, error: HTTPError
) -> str:
    """Describe a failed download without the URL, which may hold a token."""
    message = (
        f"{name}: download of {file_entry.key} failed with HTTP {error.code} "
        f"{error.reason}."
    )
    env_name = dataset.get("access_token_env")
    if env_name and error.code in {401, 403, 404}:
        where = dataset.get("source_url") or "the dataset record"
        if _read_access_token(dataset) is None:
            message += (
                f" The files may be restricted: set the environment variable "
                f"{env_name} to the token of a share link from the dataset owner, "
                f"or request access at {where}."
            )
        else:
            message += (
                f" The token in {env_name} was not accepted; check that it "
                f"belongs to a valid share link of {where}."
            )
    return message


def _request_json(url: str) -> dict:
    _validate_download_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def _zenodo_file_url(record_id: str, key: str) -> str:
    encoded_key = quote(key, safe="")
    return f"https://zenodo.org/records/{record_id}/files/{encoded_key}?download=1"


def _validate_download_url(url: str, *, allow_local_files: bool = False) -> None:
    """Require authenticated transport, except for explicit local test fixtures."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme == "https" and parsed.netloc:
        return
    if allow_local_files and scheme == "file":
        return
    raise DatasetError(
        "Dataset downloads require HTTPS URLs. Local file URLs are allowed only "
        "when allow_local_files=True."
    )


def _resolve_dataset_files(
    dataset: dict, *, allow_local_files: bool = False
) -> list[DatasetFile]:
    source = dataset.get("source", "zenodo")
    if source == "archive":
        archive_url = dataset.get("archive_url", "").strip()
        if not archive_url:
            raise DatasetError("Archive dataset is missing 'archive_url'.")
        _validate_download_url(
            archive_url,
            allow_local_files=allow_local_files,
        )
        key = dataset.get("archive_name") or Path(archive_url.split("?", 1)[0]).name
        return [
            DatasetFile(
                key=key or "dataset.archive",
                url=archive_url,
                checksum=dataset.get("checksum", ""),
            )
        ]

    if source != "zenodo":
        raise DatasetError(f"Unsupported dataset source {source!r}.")

    record_id = str(dataset.get("record_id", "")).strip()
    if not record_id:
        raise DatasetError("Zenodo dataset is missing 'record_id'.")

    record = _request_json(ZENODO_API_URL.format(record_id=record_id))
    files = []
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename")
        if not key:
            continue
        links = item.get("links", {})
        url = links.get("download") or _zenodo_file_url(record_id, key)
        _validate_download_url(url)
        files.append(
            DatasetFile(
                key=key,
                url=url,
                checksum=item.get("checksum", ""),
                size=item.get("size"),
            )
        )

    if not files:
        raise DatasetError(
            f"Zenodo record {record_id} does not list downloadable files."
        )
    return files


def _parse_checksum(checksum: str) -> tuple[str, str]:
    checksum = checksum.strip().lower()
    if not checksum:
        return "", ""
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "sha256", checksum
    return algorithm.strip(), expected.strip()


def _hexdigest(path: Path, algorithm: str) -> str:
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise DatasetError(f"Unsupported checksum algorithm {algorithm!r}.") from exc

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _verify_checksum(path: Path, checksum: str) -> None:
    algorithm, expected = _parse_checksum(checksum)
    if not algorithm:
        return
    actual = _hexdigest(path, algorithm)
    if actual != expected:
        raise DatasetError(
            f"Checksum mismatch for {path.name}: expected {algorithm}:{expected}, "
            f"got {algorithm}:{actual}."
        )


def _download_file(
    file_entry: DatasetFile,
    destination: Path,
    *,
    quiet: bool,
    allow_local_files: bool = False,
    access_token: str | None = None,
) -> None:
    _validate_download_url(
        file_entry.url,
        allow_local_files=allow_local_files,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        size = f" ({file_entry.size / 1024 / 1024:.1f} MB)" if file_entry.size else ""
        print(f"Downloading {file_entry.key}{size}")

    request = Request(
        _with_access_token(file_entry.url, access_token),
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=120) as response:
        # Zenodo answers a restricted file without a valid token with a
        # redirect to its login page instead of an HTTP error.
        if urlsplit(response.geturl()).path.startswith("/login"):
            raise HTTPError(
                file_entry.url, 401, "redirected to the login page", None, None
            )
        with destination.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _archive_member_target(
    member_name: str, target_dir: Path, target_root: Path
) -> Path:
    """Resolve an archive member path without permitting platform-specific escapes."""
    if not member_name or "\x00" in member_name:
        raise DatasetError(f"Unsafe path in archive: {member_name!r}")

    posix_path = PurePosixPath(member_name.replace("\\", "/"))
    windows_path = PureWindowsPath(member_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise DatasetError(f"Unsafe path in archive: {member_name}")

    target = (target_dir / Path(*posix_path.parts)).resolve()
    if not _is_relative_to(target, target_root):
        raise DatasetError(f"Unsafe path in archive: {member_name}")
    return target


def _zip_member_is_directory(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if member.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise DatasetError(f"Unsafe special entry in archive: {member.filename}")
        return True
    if file_type not in (0, stat.S_IFREG):
        raise DatasetError(f"Unsafe special entry in archive: {member.filename}")
    return False


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        validated_members = []
        for member in archive.infolist():
            target = _archive_member_target(
                member.filename,
                target_dir,
                target_root,
            )
            is_directory = _zip_member_is_directory(member)
            validated_members.append((member, target, is_directory))

        target_dir.mkdir(parents=True, exist_ok=True)
        for member, target, is_directory in validated_members:
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path) as archive:
        validated_members = []
        for member in archive.getmembers():
            target = _archive_member_target(member.name, target_dir, target_root)
            if not (member.isfile() or member.isdir()):
                raise DatasetError(f"Unsafe special entry in archive: {member.name}")
            validated_members.append((member, target))

        target_dir.mkdir(parents=True, exist_ok=True)
        for member, target in validated_members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise DatasetError(f"Could not read archive member: {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _extract_if_archive(path: Path, target_dir: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".zip"):
        _safe_extract_zip(path, target_dir)
        return True
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        _safe_extract_tar(path, target_dir)
        return True
    return False


def extract_archive_safely(archive_path: Path, target_dir: Path) -> None:
    """Extract a supported archive without allowing files to escape ``target_dir``.

    ZIP and TAR-family archives are validated in full before any member is
    written. Absolute paths, parent-directory traversal, links, devices, and
    other special filesystem entries are rejected.

    Parameters
    ----------
    archive_path : pathlib.Path
        ZIP or TAR-family archive to extract.
    target_dir : pathlib.Path
        Directory that will contain every extracted member.

    Raises
    ------
    DatasetError
        If the archive type is unsupported or any member is unsafe.
    """
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    if not _extract_if_archive(archive_path, target_dir):
        raise DatasetError(f"Unsupported archive format: {archive_path.name}")


def _dataset_is_present(path: Path, dataset: dict) -> bool:
    if not path.exists():
        return False
    for rel_path in dataset.get("required_paths", []):
        if not (path / rel_path).exists():
            return False
    return True


def _write_marker(
    target_dir: Path, dataset_name: str, dataset: dict, files: list[DatasetFile]
) -> None:
    marker = {
        "dataset": dataset_name,
        "record_id": dataset.get("record_id"),
        "doi": dataset.get("doi"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "key": file_entry.key,
                "checksum": file_entry.checksum,
                "size": file_entry.size,
            }
            for file_entry in files
        ],
    }
    (target_dir / ".hypermri_dataset.json").write_text(
        json.dumps(marker, indent=2) + "\n",
        encoding="utf-8",
    )


def fetch_example_data(
    name: str = "tutorial",
    *,
    data_home: str | Path | None = None,
    registry_path: str | Path | None = None,
    force: bool = False,
    quiet: bool = False,
    allow_local_files: bool = False,
) -> Path:
    """Download an approved example dataset if needed and return its local path.

    The packaged registry names each released dataset, its download URL and
    the checksum of the file. A registry entry with ``access_token_env``
    accepts a Zenodo share-link token from that environment variable
    (``HYPERMRI_ZENODO_TOKEN`` for ``"hp_pyruvate_csi_mouse"``); the token
    opens restricted files and is never printed or written to the cache.
    Without the variable the download is attempted without a token. The first
    call downloads the file into
    ``<data_home>/.downloads``, verifies the checksum, unpacks archives safely
    and writes a marker file ``.hypermri_dataset.json`` with the source and
    the download time. Later calls find the required files and return at once
    without network access.

    Parameters
    ----------
    name : str
        Registry name of the dataset, for example
        ``"hp_pyruvate_csi_mouse"``. :func:`list_datasets` returns the names
        that can be downloaded. The default ``"tutorial"`` is a reserved
        placeholder that is not released.
    data_home : str or pathlib.Path, optional
        Cache root. Defaults to the environment variable ``HYPERMRI_DATA`` or,
        without it, to the user cache folder of the operating system.
    registry_path : str or pathlib.Path, optional
        Registry JSON file to use instead of the packaged one.
    force : bool, default=False
        Download and unpack again even if the dataset is already present.
    quiet : bool, default=False
        Suppress the progress messages.
    allow_local_files : bool, default=False
        Accept ``file://`` URLs. Intended only for isolated tests with local
        fixtures; registry downloads otherwise require HTTPS.

    Returns
    -------
    pathlib.Path
        Folder of the usable dataset, for a scanner session the folder to
        pass to :class:`hypermri.Study`.

    Raises
    ------
    DatasetError
        If the name is unknown or not released (checked before any file or
        network access), if the server refuses the download (the message
        names the token variable for restricted files), if a checksum does
        not match, if an archive member is unsafe, or if required files are
        missing after unpacking.

    Examples
    --------
    >>> from hypermri import Study
    >>> from hypermri.datasets import fetch_example_data
    >>> path = fetch_example_data("hp_pyruvate_csi_mouse")  # doctest: +SKIP
    >>> study = Study(path)  # doctest: +SKIP
    """
    registry = load_registry(registry_path)
    dataset = get_dataset_info(name, registry)
    _require_dataset_available(name, dataset)
    root = get_data_home(data_home, registry)
    target_path = dataset_path(root, name, dataset)
    target_dir = target_path.parent if dataset.get("relative_path") else target_path

    if _dataset_is_present(target_path, dataset) and not force:
        if not quiet:
            print(f"{name}: already available at {target_path}")
        return target_path

    access_token = _read_access_token(dataset)
    target_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = target_dir / ".downloads"
    files = _resolve_dataset_files(
        dataset,
        allow_local_files=allow_local_files,
    )

    for file_entry in files:
        destination = _archive_member_target(
            file_entry.key,
            downloads_dir,
            downloads_dir.resolve(),
        )
        if force or not destination.exists():
            try:
                _download_file(
                    file_entry,
                    destination,
                    quiet=quiet,
                    allow_local_files=allow_local_files,
                    access_token=access_token,
                )
            except HTTPError as exc:
                # ``from None``: the chained error would show the URL and token.
                raise DatasetError(
                    _refused_download_message(name, dataset, file_entry, exc)
                ) from None
        _verify_checksum(destination, file_entry.checksum)

        if _extract_if_archive(destination, target_dir):
            continue
        output_path = _archive_member_target(
            file_entry.key,
            target_dir,
            target_dir.resolve(),
        )
        if force or not output_path.exists():
            output_path.write_bytes(destination.read_bytes())

    if not _dataset_is_present(target_path, dataset):
        required = ", ".join(dataset.get("required_paths", [])) or "dataset files"
        raise DatasetError(
            f"{name}: download finished, but expected files were not found under "
            f"{target_path}. Required: {required}."
        )

    _write_marker(target_dir, name, dataset, files)
    if not quiet:
        print(f"{name}: ready at {target_path}")
    return target_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="tutorial",
        help="Dataset name to fetch. Defaults to 'tutorial'.",
    )
    parser.add_argument(
        "--data-home",
        help="Dataset cache root. Defaults to HYPERMRI_DATA or the user cache.",
    )
    parser.add_argument(
        "--registry",
        help="Use a custom registry JSON file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract again even if required files are present.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        registry = load_registry(args.registry)
        if args.list:
            available = list_datasets(registry)
            if not available:
                print("No public example datasets are currently released.")
                return 0
            for dataset_name in available:
                info = get_dataset_info(dataset_name, registry)
                print(f"{dataset_name}\t{info.get('description', '')}")
            return 0

        path = fetch_example_data(
            args.dataset,
            data_home=args.data_home,
            registry_path=args.registry,
            force=args.force,
            quiet=args.quiet,
        )
    except DatasetError as exc:
        parser.exit(1, f"error: {exc}\n")

    if args.quiet:
        print(path)
    return 0
