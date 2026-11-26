"""Global pytest fixtures and synthetic test data generators."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Optional

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def _isolate_from_running_git_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the variables Git exports while it runs a hook.

    The commit hook runs the test suite with ``GIT_INDEX_FILE``, ``GIT_DIR``
    and similar variables set for the commit in progress. A test that runs
    ``git add`` in a temporary repository would otherwise write into that
    commit's index. Removing them also makes tests behave the same inside and
    outside a hook.
    """
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)


@pytest.fixture
def package_subprocess_env() -> dict[str, str]:
    """Keep child Python imports on the package selected by the parent test.

    Returns
    -------
    dict of str to str
        Copy of the current environment with ``PYTHONPATH`` pointing to the
        active HyperMRI package's import root. Source-checkout tests retain
        their selected source tree; installed-wheel tests retain their installed
        package instead of switching to the checkout's ``src`` directory.

    Notes
    -----
    Importing the lazy package root reads no acquired data and initializes no
    numerical or UI modules. Each request returns a fresh mutable environment
    without changing the parent process or starting a subprocess.
    """
    import hypermri

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(hypermri.__file__).resolve().parent.parent)
    return environment


# ---------------------------------------------------------------------------
# Optional real-data fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_MANIFEST = PROJECT_ROOT / "src" / "hypermri" / "datasets" / "registry.json"
DEFAULT_REAL_DATA_ROOT = PROJECT_ROOT / "tests" / "data"
REAL_DATA_PROVENANCE_FILENAME = ".hypermri-test-data.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the mandatory reference-data mode for integration CI.

    Parameters
    ----------
    parser : pytest.Parser
        Active pytest command-line parser. Adds ``--require-real-data`` to
        replace availability skips from ``real_dataset_path`` with failures.

    Returns
    -------
    None
        Mutates the parser only; no datasets are read or downloaded.
    """
    parser.getgroup("real data").addoption(
        "--require-real-data",
        action="store_true",
        default=False,
        help="Fail instead of skipping unavailable or unverified reference datasets.",
    )


def _load_real_data_manifest() -> dict:
    from hypermri.datasets import load_registry

    registry = load_registry(REAL_DATA_MANIFEST)
    return {
        "version": registry["version"],
        "data_root_env": registry["test_data_root_env"],
        "datasets": registry["reference_datasets"],
    }


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            prefix = stream.read(128)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


@pytest.fixture(scope="session")
def real_data_manifest() -> dict:
    """Return the external real-data manifest."""
    return _load_real_data_manifest()


@pytest.fixture(scope="session")
def real_data_root(real_data_manifest: dict) -> Path:
    """Return the local real-data cache root."""
    env_name = real_data_manifest.get("data_root_env", "HYPERMRI_TEST_DATA")
    root = os.environ.get(env_name)
    if root:
        return Path(root).expanduser().resolve()
    return DEFAULT_REAL_DATA_ROOT


@pytest.fixture(scope="session")
def real_dataset_path(
    real_data_manifest: dict,
    real_data_root: Path,
    pytestconfig: pytest.Config,
) -> Callable[[str], Path]:
    """Resolve verified reference data with optional or mandatory availability.

    Parameters
    ----------
    real_data_manifest : dict
        Reference dataset IDs, relative paths, required files and pinned
        archive identities from the shared registry.
    real_data_root : pathlib.Path
        Root of the external or repository-local reference-data cache.
    pytestconfig : pytest.Config
        Active test configuration. ``--require-real-data`` turns missing,
        disabled, unreleased or unverified datasets into test failures.

    Returns
    -------
    callable
        Resolver accepting a dataset ID and returning its verified directory.
        Resolution checks required paths, LFS pointer prefixes and the JSON
        provenance marker; it never downloads or mutates data. Creating the
        resolver does not read dataset files.

    Raises
    ------
    pytest.fail.Exception
        When resolving an unknown ID, or when a required dataset is unavailable.
    pytest.skip.Exception
        When an optional dataset is unavailable and mandatory mode is off.
    """

    def _unavailable(reason: str) -> NoReturn:
        if pytestconfig.getoption("--require-real-data"):
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)

    def resolve(dataset_name: str) -> Path:
        datasets = real_data_manifest.get("datasets", {})
        if dataset_name not in datasets:
            pytest.fail(f"Unknown real-data dataset: {dataset_name}")

        dataset = datasets[dataset_name]
        release_status = dataset.get("release_status", "released")
        if release_status != "released" or dataset.get("enabled", True) is False:
            reason = dataset.get(
                "disabled_reason",
                "The dataset has not been approved for use.",
            )
            _unavailable(
                f"Real-data dataset {dataset_name!r} is {release_status}: {reason}"
            )

        path = real_data_root / dataset["relative_path"]
        missing_paths = [
            path / rel_path
            for rel_path in dataset.get("required_paths", [])
            if not (path / rel_path).exists() or _is_lfs_pointer(path / rel_path)
        ]
        if not path.exists() or missing_paths:
            env_name = real_data_manifest.get("data_root_env", "HYPERMRI_TEST_DATA")
            _unavailable(
                f"Real-data dataset {dataset_name!r} not available at {path}. "
                f"Set {env_name} or run scripts/download_test_data.py."
            )
        marker_path = path / REAL_DATA_PROVENANCE_FILENAME
        try:
            provenance = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = None
        expected_checksum = dataset.get("checksum", "").strip().lower()
        if not expected_checksum and dataset.get("sha256", "").strip():
            expected_checksum = f"sha256:{dataset['sha256'].strip().lower()}"
        expected_provenance = {
            "schema_version": 1,
            "dataset_id": dataset_name,
            "relative_path": dataset["relative_path"],
            "archive_name": dataset.get("archive_name", ""),
            "archive_size_bytes": dataset.get("archive_size_bytes"),
            "checksum": expected_checksum,
        }
        if provenance != expected_provenance:
            _unavailable(
                f"Real-data dataset {dataset_name!r} at {path} has no valid "
                "HyperMRI provenance marker. Run scripts/download_test_data.py "
                "to verify and materialize the pinned archive."
            )
        return path

    return resolve


# ---------------------------------------------------------------------------
# ParaVision parameter files and image comparison
# ---------------------------------------------------------------------------

JCAMP_HEADER = "##TITLE=Parameter List\n##JCAMPDX=4.24\n##DATATYPE=Parameter Values\n"
SPATIAL_INDEX_DIMS = ("read_idx", "phase_idx", "slice_idx")


def _format_jcamp_value(value) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"Cannot write {value!r} as a ParaVision value")
    return repr(value) if isinstance(value, float) else str(value)


def _format_jcamp_parameters(parameters: dict) -> str:
    lines = []
    for name, value in parameters.items():
        if value is None:
            continue
        if isinstance(value, list):
            array = np.asarray(value)
            shape = ", ".join(str(size) for size in array.shape)
            lines.append(f"##${name}=( {shape} )")
            lines.append(
                " ".join(_format_jcamp_value(v) for v in array.ravel().tolist())
            )
        else:
            lines.append(f"##${name}={_format_jcamp_value(value)}")
    lines.append("##END=")
    return JCAMP_HEADER + "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def write_paravision_parameters() -> Callable[[Path, dict], None]:
    """Return a writer of parsed ParaVision parameters as a JCAMP-DX file.

    The writer takes the file path and a mapping of parameter names to values
    as ``BrukerHandler`` parses them.  Lists are written as arrays with their
    shape, scalars and strings inline, and ``None`` (an absent parameter) is
    left out.  Parent directories are created.
    """

    def write(path: Path, parameters: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_format_jcamp_parameters(parameters), encoding="utf-8")

    return write


_SYNTHETIC_FLASH_VISU_PARS = {
    "VisuCoreSize": [2, 2],
    "VisuCreatorVersion": "7.0.0",
    "VisuCoreWordType": "_16BIT_SGN_INT",
    "VisuCoreByteOrder": "littleEndian",
    "VisuCoreFrameCount": 2,
    "VisuCoreDimDesc": ["spatial", "spatial"],
    "VisuFGOrderDescDim": 1,
    "VisuFGOrderDesc": ["(2, <FG_SLICE>, <>, 0, 0)"],
    "VisuCoreDataSlope": [1, 1],
    "VisuCoreDataOffs": [0, 0],
    "VisuCoreFrameType": "MAGNITUDE_IMAGE",
}


@pytest.fixture
def synthetic_flash_scan(tmp_path, write_paravision_parameters) -> Path:
    """Write a synthetic two-slice ParaVision FLASH series without raw data.

    The series has two 2 x 2 slices of 16-bit integers 0..7 in ``2dseq``,
    the method name ``<Bruker:FLASH>``, an identity ``ACQ_grad_matrix`` per
    slice (present in every ParaVision ``acqp``; the scanner-image
    reconstruction reads it), and no other geometry and no ``fid`` file.
    """
    scan = tmp_path / "scan"
    write_paravision_parameters(scan / "method", {"Method": "<Bruker:FLASH>"})
    write_paravision_parameters(
        scan / "acqp",
        {
            "ACQ_scan_name": "<synthetic>",
            "ACQ_grad_matrix": [np.eye(3, dtype=int).tolist()] * 2,
        },
    )
    write_paravision_parameters(scan / "visu_pars", _SYNTHETIC_FLASH_VISU_PARS)
    write_paravision_parameters(
        scan / "pdata" / "1" / "visu_pars", _SYNTHETIC_FLASH_VISU_PARS
    )
    write_paravision_parameters(
        scan / "pdata" / "1" / "reco", {"RECO_transposition": [0, 0]}
    )
    (scan / "pdata" / "1" / "2dseq").write_bytes(np.arange(8, dtype="<i2").tobytes())
    return scan


def _spatial_magnitude(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return np.abs(image).astype(np.float64)
    extra_dims = [dim for dim in image.dims if dim not in SPATIAL_INDEX_DIMS]
    non_singleton = {
        dim: image.sizes[dim] for dim in extra_dims if image.sizes[dim] != 1
    }
    if non_singleton:
        raise ValueError(
            f"Cannot compare spatial magnitudes: non-spatial dimensions "
            f"{non_singleton} have more than one entry; select one entry first"
        )
    spatial = image.squeeze(extra_dims, drop=True).transpose(*SPATIAL_INDEX_DIMS)
    return np.abs(spatial.to_numpy()).astype(np.float64)


@pytest.fixture(scope="session")
def magnitude_cross_correlation() -> Callable:
    """Return the normalised circular cross-correlation of two magnitude images.

    The returned function takes two images and returns
    ``(zero_shift, peak_shift)``.  Each image is either an ``xarray`` object
    with the dimensions ``read_idx``, ``phase_idx`` and ``slice_idx`` plus
    singleton dimensions, or an array already ordered as
    ``(read, phase, slice)``.  Both are reduced to float64 magnitudes with
    their means removed.  ``zero_shift`` is the correlation without a shift,
    1 for identical images up to a positive scale factor.  ``peak_shift`` is
    the signed circular shift in voxels per axis at which the correlation is
    largest, ``(0, 0, 0)`` when the images are aligned.  A swapped, mirrored
    or shifted axis of an asymmetric image lowers ``zero_shift`` or moves the
    peak.
    """

    def correlate(first, second) -> tuple[float, tuple[int, ...]]:
        a = _spatial_magnitude(first)
        b = _spatial_magnitude(second)
        if a.shape != b.shape:
            raise ValueError(
                f"Cannot correlate magnitudes of shapes {a.shape} and {b.shape}"
            )
        a = a - a.mean()
        b = b - b.mean()
        correlation = np.real(
            np.fft.ifftn(np.fft.fftn(a) * np.conj(np.fft.fftn(b)))
        ) / (np.linalg.norm(a) * np.linalg.norm(b))
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_shift = tuple(
            int(index if index <= size // 2 else index - size)
            for index, size in zip(peak, correlation.shape, strict=True)
        )
        return float(correlation.flat[0]), peak_shift

    return correlate


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_anatomy_data(
    read: int = 64,
    phase: int = 48,
    slices: int = 12,
    channels: int = 1,
    repetitions: int = 1,
    echoes: int = 1,
    dtype: type = np.float32,
    seed: int = 42,
    slice_thickness_mm: float = 1.0,
    fov_mm: tuple[float, ...] = (30.0, 24.0, 12.0),
    orientation: str = "axial",
) -> xr.DataArray:
    """Create a synthetic anatomical MRI DataArray.

    Parameters
    ----------
    read, phase, slices : int
        Spatial matrix dimensions.
    channels : int
        Number of receiver channels.
    repetitions : int
        Number of repetitions (dynamics / time points).
    echoes : int
        Number of echoes.
    dtype : numpy dtype
        Data type for the array values.
    seed : int
        RNG seed for reproducibility.
    slice_thickness_mm : float
        Slice thickness in millimetres.
    fov_mm : tuple of float
        Field-of-view in (read, phase, slice) directions.
    orientation : str
        Slice orientation label ("axial", "coronal", or "sagittal").

    Returns
    -------
    xr.DataArray
        Anatomical DataArray with standard HyperMRI dimensions,
        coordinates, and attributes.
    """
    rng = np.random.default_rng(seed)

    shape = (read, phase, slices, channels, repetitions, echoes)
    dims = [
        "read_idx",
        "phase_idx",
        "slice_idx",
        "channel_idx",
        "repetition_idx",
        "echo_idx",
    ]

    data = rng.standard_normal(shape).astype(dtype)

    resolution_mm = [fov_mm[i] / s for i, s in enumerate([read, phase, slices])]

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "channel_idx": np.arange(channels),
        "repetition_idx": np.arange(repetitions),
        "echo_idx": np.arange(echoes),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    # Attach coordinate metadata matching recon_strategy conventions
    for i, dim in enumerate(["read_idx", "phase_idx", "slice_idx"]):
        da[dim].attrs["matrix"] = shape[i]
        da[dim].attrs["resolution_mm"] = resolution_mm[i]
        da[dim].attrs["extent_mm"] = fov_mm[i]
        da[dim].attrs["axis_mm"] = np.linspace(-fov_mm[i] / 2, fov_mm[i] / 2, shape[i])

    # Standard DataArray attributes
    da.attrs["domain"] = "spatial"
    da.attrs["is_spectroscopic"] = False
    da.attrs["slice_orientation"] = orientation
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["slice_thickness_mm"] = slice_thickness_mm
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    # Squeeze singleton dims so callers get the shape they expect
    da = da.squeeze(drop=False)

    return da


def make_spectroscopy_data(
    read: int = 1,
    phase: int = 1,
    slices: int = 1,
    channels: int = 1,
    acquisitions: int = 2048,
    repetitions: int = 1,
    dtype: type = np.complex64,
    seed: int = 42,
    dwell_time_s: float = 100e-6,
    bandwidth_hz: float = 5000.0,
    center_ppm: float = 0.0,
) -> xr.DataArray:
    """Create a synthetic spectroscopy (FID) DataArray.

    Parameters
    ----------
    read, phase, slices : int
        Spatial voxel counts (typically 1 for single-voxel spectroscopy).
    channels : int
        Number of receiver channels.
    acquisitions : int
        Number of acquisition points (FID length).
    repetitions : int
        Number of repetitions / averages.
    dtype : numpy dtype
        Data type (complex for FID data).
    seed : int
        RNG seed.
    dwell_time_s : float
        Dwell time in seconds.
    bandwidth_hz : float
        Spectral bandwidth in Hz.
    center_ppm : float
        Center frequency in ppm.

    Returns
    -------
    xr.DataArray
        Spectroscopy DataArray with standard HyperMRI structure.
    """
    rng = np.random.default_rng(seed)

    shape = (read, phase, slices, channels, acquisitions, repetitions)
    dims = [
        "read_idx",
        "phase_idx",
        "slice_idx",
        "channel_idx",
        "acquisition_idx",
        "repetition_idx",
    ]

    # Simulate a decaying FID with noise
    t = np.arange(acquisitions) * dwell_time_s
    fid_base = np.exp(-t / 0.05) * np.exp(1j * 2 * np.pi * 50 * t)
    data = np.zeros(shape, dtype=dtype)
    for idx in np.ndindex(*shape[:4]):
        noise = rng.standard_normal(acquisitions) + 1j * rng.standard_normal(
            acquisitions
        )
        data[idx] = (fid_base + 0.1 * noise).astype(dtype)
    # Broadcast across repetitions
    if repetitions > 1:
        data = np.broadcast_to(data[..., :1], shape).copy()  # copy to make writeable

    sampling_time_axis_s = t

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "channel_idx": np.arange(channels),
        "acquisition_idx": np.arange(acquisitions),
        "repetition_idx": np.arange(repetitions),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    # Add acquisition time coordinate
    da = da.assign_coords(
        acquisition_time_s=(
            "acquisition_idx",
            sampling_time_axis_s,
            {"unit": "s", "long_name": "Acquisition time"},
        )
    )

    # Attributes
    da.attrs["domain"] = "kspace"
    da.attrs["signal_domain"] = "temporal"
    da.attrs["is_spectroscopic"] = True
    da.attrs["slice_orientation"] = "axial"
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["dwell_time_s"] = dwell_time_s
    da.attrs["bandwidth_hz"] = bandwidth_hz
    da.attrs["center_ppm"] = center_ppm
    da.attrs["sampling_time_axis_s"] = sampling_time_axis_s
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    da = da.squeeze(drop=False)

    return da


def make_dce_data(
    read: int = 32,
    phase: int = 32,
    slices: int = 8,
    repetitions: int = 60,
    frame_time_s: float = 3.0,
    n_precontrast: int = 5,
    dtype: type = np.float32,
    seed: int = 42,
) -> xr.DataArray:
    """Create synthetic DCE (dynamic contrast-enhanced) MRI data.

    Simulates a signal-vs-time curve with a pre-contrast baseline
    followed by contrast agent uptake (gamma-variate bolus).

    Parameters
    ----------
    read, phase, slices : int
        Spatial matrix dimensions.
    repetitions : int
        Number of time frames.
    frame_time_s : float
        Time between frames in seconds.
    n_precontrast : int
        Number of pre-contrast baseline frames.
    dtype : numpy dtype
        Data type.
    seed : int
        RNG seed.

    Returns
    -------
    xr.DataArray
        DCE DataArray with time coordinate on repetition_idx.
    """
    rng = np.random.default_rng(seed)

    shape = (read, phase, slices, repetitions)
    dims = ["read_idx", "phase_idx", "slice_idx", "repetition_idx"]

    # Build a gamma-variate-like bolus curve
    t = np.arange(repetitions) * frame_time_s
    baseline = 1000.0
    bolus_t = t - n_precontrast * frame_time_s
    bolus_t = np.clip(bolus_t, 0, None)
    bolus = 500.0 * bolus_t * np.exp(-bolus_t / 20.0)
    signal_curve = baseline + bolus

    data = np.ones(shape, dtype=dtype)
    data *= signal_curve[np.newaxis, np.newaxis, np.newaxis, :]
    data += rng.standard_normal(shape).astype(dtype) * 20.0

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "repetition_idx": np.arange(repetitions),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    # Time coordinate
    time_coords = t.astype(np.float64)
    da = da.assign_coords(time=("repetition_idx", time_coords))
    da["time"].attrs["unit"] = "s"
    da["time"].attrs["long_name"] = "Frame acquisition time"

    da.attrs["domain"] = "spatial"
    da.attrs["is_spectroscopic"] = False
    da.attrs["slice_orientation"] = "axial"
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    return da


def make_multiecho_data(
    read: int = 64,
    phase: int = 48,
    slices: int = 12,
    echoes: int = 8,
    echo_spacing_ms: float = 5.0,
    t2_star_ms: float = 25.0,
    dtype: type = np.float32,
    seed: int = 42,
) -> xr.DataArray:
    """Create synthetic multi-echo gradient-echo data (e.g. for T2* mapping).

    Parameters
    ----------
    read, phase, slices : int
        Spatial matrix dimensions.
    echoes : int
        Number of echoes.
    echo_spacing_ms : float
        Echo spacing in milliseconds.
    t2_star_ms : float
        Simulated T2* decay constant.
    dtype : numpy dtype
        Data type.
    seed : int
        RNG seed.

    Returns
    -------
    xr.DataArray
        Multi-echo DataArray with echo_idx dimension.
    """
    rng = np.random.default_rng(seed)

    shape = (read, phase, slices, echoes)
    dims = ["read_idx", "phase_idx", "slice_idx", "echo_idx"]

    te_ms = np.arange(1, echoes + 1) * echo_spacing_ms
    decay = np.exp(-te_ms / t2_star_ms)

    data = np.ones(shape, dtype=dtype) * 1000.0
    data *= decay[np.newaxis, np.newaxis, np.newaxis, :]
    data += rng.standard_normal(shape).astype(dtype) * 10.0

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "echo_idx": np.arange(echoes),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    da.attrs["domain"] = "spatial"
    da.attrs["is_spectroscopic"] = False
    da.attrs["slice_orientation"] = "axial"
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["echo_spacing_ms"] = echo_spacing_ms
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    return da


def make_inversion_recovery_data(
    read: int = 48,
    phase: int = 48,
    slices: int = 5,
    n_inversions: int = 10,
    t1_ms: float = 1200.0,
    dtype: type = np.float32,
    seed: int = 42,
) -> xr.DataArray:
    """Create synthetic inversion-recovery data for T1 mapping.

    Parameters
    ----------
    read, phase, slices : int
        Spatial matrix dimensions.
    n_inversions : int
        Number of inversion times.
    t1_ms : float
        Simulated T1 in milliseconds.
    dtype : numpy dtype
        Data type.
    seed : int
        RNG seed.

    Returns
    -------
    xr.DataArray
        Inversion-recovery DataArray with inversion_idx dimension
        and 2D TI coordinate (slice_idx, inversion_idx).
    """
    rng = np.random.default_rng(seed)

    shape = (read, phase, slices, n_inversions)
    dims = ["read_idx", "phase_idx", "slice_idx", "inversion_idx"]

    # Log-spaced inversion times from 50 to 5000 ms
    ti_ms = np.logspace(np.log10(50), np.log10(5000), n_inversions)

    # IR signal: M0 * (1 - 2*exp(-TI/T1))
    signal = 1000.0 * np.abs(1 - 2 * np.exp(-ti_ms / t1_ms))

    data = np.ones(shape, dtype=dtype)
    data *= signal[np.newaxis, np.newaxis, np.newaxis, :]
    data += rng.standard_normal(shape).astype(dtype) * 10.0

    # 2D TI coordinate: (slice_idx, inversion_idx)
    # Each slice can have slightly different effective TI
    ti_2d = np.tile(ti_ms, (slices, 1))

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "inversion_idx": (
            ("slice_idx", "inversion_idx"),
            ti_2d,
        ),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    da.attrs["domain"] = "spatial"
    da.attrs["is_spectroscopic"] = False
    da.attrs["slice_orientation"] = "axial"
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    return da


def make_diffusion_data(
    read: int = 48,
    phase: int = 48,
    slices: int = 8,
    b_values: Optional[list[float]] = None,
    adc_mm2_s: float = 1.0e-3,
    dtype: type = np.float32,
    seed: int = 42,
) -> xr.DataArray:
    """Create synthetic diffusion-weighted imaging data.

    Parameters
    ----------
    read, phase, slices : int
        Spatial matrix dimensions.
    b_values : list of float, optional
        B-values in s/mm^2. Defaults to [0, 200, 500, 800, 1000].
    adc_mm2_s : float
        Apparent diffusion coefficient in mm^2/s.
    dtype : numpy dtype
        Data type.
    seed : int
        RNG seed.

    Returns
    -------
    xr.DataArray
        Diffusion DataArray with bvalue_idx dimension.
    """
    if b_values is None:
        b_values = [0.0, 200.0, 500.0, 800.0, 1000.0]
    rng = np.random.default_rng(seed)

    n_bvals = len(b_values)
    shape = (read, phase, slices, n_bvals)
    dims = ["read_idx", "phase_idx", "slice_idx", "bvalue_idx"]

    b_arr = np.array(b_values)
    signal = 1000.0 * np.exp(-b_arr * adc_mm2_s)

    data = np.ones(shape, dtype=dtype)
    data *= signal[np.newaxis, np.newaxis, np.newaxis, :]
    data += rng.standard_normal(shape).astype(dtype) * 10.0

    coords: dict = {
        "read_idx": np.arange(read),
        "phase_idx": np.arange(phase),
        "slice_idx": np.arange(slices),
        "bvalue_idx": np.arange(n_bvals),
    }

    da = xr.DataArray(data, dims=dims, coords=coords)

    da.attrs["domain"] = "spatial"
    da.attrs["is_spectroscopic"] = False
    da.attrs["slice_orientation"] = "axial"
    da.attrs["read_orientation"] = "H_F"
    da.attrs["patient_pos"] = "Head_Prone"
    da.attrs["n_slices"] = slices
    da.attrs["b_values"] = b_values
    da.attrs["unit"] = "a.u."
    da.attrs["num_image_fov_shift"] = 0
    da.attrs["read_direction"] = np.array([1.0, 0.0, 0.0])
    da.attrs["phase_direction"] = np.array([0.0, 1.0, 0.0])
    da.attrs["slice_direction"] = np.array([0.0, 0.0, 1.0])

    return da
