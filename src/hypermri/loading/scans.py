"""
Factory module for creating MRI experiments.

This module provides factory functions for creating MRI experiments
with automatic vendor detection and appropriate class selection.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Type, Union

import numpy as np

from hypermri.exceptions import DataLoadingError
from hypermri.loading.experiment import MRIExperiment, _handler_vendor_name
from hypermri.loading.protocols import VendorHandlerProtocol
from hypermri.vendors import get_vendor_handler, get_vendor_registry

logger = logging.getLogger(__name__)

LoadMode = Literal["lazy", "minimal", "full"]


def normalize_load_mode(mode: bool | str) -> LoadMode:
    """Return the canonical loading mode used by scan-loading workflows.

    Boolean values remain supported for compatibility with
    :func:`create_experiment`: ``False`` means metadata-only/lazy loading and
    ``True`` means full loading. String modes make the intended amount of work
    explicit.

    Parameters
    ----------
    mode
        One of ``False``, ``True``, ``"lazy"``, ``"minimal"``, or ``"full"``.

    Returns
    -------
    {"lazy", "minimal", "full"}
        Normalized loading mode.

    Raises
    ------
    TypeError
        If ``mode`` is neither a boolean nor a string.
    ValueError
        If a string does not name a supported loading mode.
    """
    if mode is True:
        return "full"
    if mode is False:
        return "lazy"
    if not isinstance(mode, str):
        raise TypeError(
            "Loading mode must be a boolean or one of 'lazy', 'minimal', and 'full'."
        )

    normalized = mode.strip().lower()
    if normalized == "lazy":
        return "lazy"
    if normalized == "minimal":
        return "minimal"
    if normalized == "full":
        return "full"
    raise ValueError(
        f"Unsupported loading mode {mode!r}. Choose 'lazy', 'minimal', or 'full'."
    )


def _resolve_sequence_class(method_name: str, vendor: str | None = None):
    """Return the registered sequence class for the given method name and vendor.

    Uses Python's module import cache, so repeated calls are inexpensive.
    """
    entry = _resolve_sequence_entry(method_name, vendor)
    if entry:
        return entry["sequence_class"]
    return None


def _resolve_sequence_entry(method_name: str, vendor: str | None = None):
    """Return the registered sequence entry for the given method and vendor."""
    from hypermri.sequences.registry import get_sequence_registry

    return get_sequence_registry().resolve_sequence(method_name, vendor)


def _get_sequence_vendor(
    handler: VendorHandlerProtocol, force_vendor: str | None = None
) -> str | None:
    """Return the vendor key used to resolve the sequence class of a handler."""
    if force_vendor:
        return force_vendor.strip().lower()

    vendor = _handler_vendor_name(handler)
    return None if vendor == "unknown" else vendor


def _apply_sequence_load_policy(
    sequence_obj: MRIExperiment,
    load_data: bool | str,
    *,
    category=None,
    reco_options: dict[str, Any] | None = None,
) -> MRIExperiment:
    """Run central sequence reconstruction policy when available."""
    if hasattr(sequence_obj, "run_default_reconstructions"):
        sequence_obj.run_default_reconstructions(
            load_data=load_data,
            reco_options=reco_options or {},
            category=category,
        )
    return sequence_obj


def _load_handler_data(experiment: MRIExperiment) -> None:
    """Load a vendor payload and mirror its primary array onto the experiment."""
    handler = experiment.handler

    def cached_array(public_name: str):
        private_name = f"_{public_name}"
        if hasattr(handler, private_name):
            return getattr(handler, private_name)
        return getattr(handler, public_name, None)

    handler_reco = cached_array("data")
    handler_scanner = cached_array("scanner_data")
    if handler_reco is None and handler_scanner is None:
        handler.load_data(getattr(handler, "path", experiment.path))
        handler_reco = cached_array("data")
        handler_scanner = cached_array("scanner_data")

    if handler_reco is not None:
        experiment.set_primary_data(handler_reco, "data")
        experiment.reconstructions["raw"] = handler_reco
    elif handler_scanner is not None:
        experiment.set_primary_data(handler_scanner, "scanner_data")


def _prepare_minimal_sequence_data(
    experiment: MRIExperiment,
    load_mode: LoadMode,
) -> None:
    """Prepare minimal data for vendors without a seq2d-style loader."""
    handler = getattr(experiment, "handler", None)
    if load_mode != "minimal" or handler is None:
        return
    if hasattr(handler, "load_2dseq_file"):
        return

    _load_handler_data(experiment)


def _apply_base_load_policy(
    experiment: MRIExperiment,
    load_mode: LoadMode,
) -> MRIExperiment:
    """Preserve minimal loading for scans without a registered sequence class."""
    if load_mode != "minimal":
        return experiment

    handler = getattr(experiment, "handler", None)
    if handler is None:
        return experiment

    if hasattr(handler, "load_2dseq_file"):
        experiment.reconstruct("seq2d")
    else:
        _load_handler_data(experiment)

    return experiment


def load_scan(
    path: str | Path,
    mode: bool | str = "lazy",
    sequence_class: type | None = None,
    force_vendor: str | None = None,
    handler_options: dict[str, Any] | None = None,
    log_mode: str = "error",
    resolve_sequence: bool = True,
) -> MRIExperiment:
    """Load one scan and return its registered sequence representation.

    This is the canonical entry point for loading an individual scan. It
    detects the vendor and constructs the vendor handler once, then constructs
    the registered sequence class directly from that handler; a scan without a
    registered class becomes a generic ``MRIExperiment`` of the same handler.
    An unresolved method is logged at warning level unless sequence resolution
    is disabled or an explicit class is supplied. Registry import errors propagate.
    Vendor-handler loading stays separate from the sequence reconstruction
    policy: only ``"full"`` asks a handler to load data during construction,
    while ``"minimal"`` starts from metadata and lets the resolved sequence run
    its minimal reconstruction policy.

    Parameters
    ----------
    path
        Filesystem location of the experiment. Relative paths are resolved
        against the current working directory.
    mode
        Loading mode. ``"lazy"`` reads metadata without automatic
        reconstruction, ``"minimal"`` runs the sequence's minimal
        reconstruction policy, and ``"full"`` requests eager vendor loading
        plus the full sequence policy. Booleans remain accepted: ``False`` maps
        to ``"lazy"`` and ``True`` maps to ``"full"``.
    sequence_class
        Optional class called with the scan's vendor handler instead of the
        registered sequence class. The class must accept the handler as its
        only positional argument.
    force_vendor
        Optional vendor string that bypasses automatic vendor detection.
    handler_options
        Additional keyword arguments passed to the vendor handler factory.
    log_mode
        Logging threshold passed to the experiment and vendor handler.
    resolve_sequence
        If true, construct the scan's registered sequence class when one is
        available. If false, return a generic ``MRIExperiment`` while still
        applying the requested loading mode.

    Returns
    -------
    MRIExperiment
        Registered sequence object, ``sequence_class`` instance, or generic
        experiment owning the detected or forced vendor handler.

    Raises
    ------
    TypeError
        If ``mode`` has an unsupported type or ``resolve_sequence`` is not a
        boolean.
    ValueError
        If ``mode`` names an unsupported loading mode.
    DataLoadingError
        If ``path`` does not exist, no registered vendor recognizes it, or
        ``force_vendor`` names no registered vendor; also raised by vendor
        handlers that recognize a scan but cannot read it.
    ReconstructionError
        If a reconstruction requested by the loading mode fails, for a
        registered sequence and for a generic ``MRIExperiment`` alike; no
        scan is returned.
    Exception
        Errors of the vendor handler's parameter reading, of the sequence
        constructor and of minimal vendor-data loading propagate unchanged;
        the scan is never returned without its data or as a generic
        ``MRIExperiment`` in their place.
    ImportError
        If the sequence registry cannot be imported; this is a configuration
        or installation defect rather than an unregistered scanner method.
    """
    load_mode = normalize_load_mode(mode)
    if not isinstance(resolve_sequence, bool):
        raise TypeError("resolve_sequence must be a boolean.")
    handler_load_data = load_mode == "full"

    # Construct the vendor handler once. String modes must never be passed to
    # handlers because both "lazy" and "minimal" are truthy Python values.
    handler = get_vendor_handler(
        source=str(path),
        load_data=handler_load_data,
        log_mode=log_mode,
        force_vendor=force_vendor,
        handler_options=handler_options or {},
    )
    if handler is None:
        _raise_unrecognized_scan(path, force_vendor)

    sequence_entry = None
    if resolve_sequence and not sequence_class:
        vendor = _get_sequence_vendor(handler, force_vendor)
        if vendor:
            sequence_entry = _resolve_sequence_entry(handler.type, vendor)

    if sequence_entry:
        reco_options = {"minimal_loading": load_mode == "minimal"}
        sequence_obj = sequence_entry["sequence_class"](
            handler,
            path=path,
            load_data=load_mode,
            reco_options=reco_options,
            log_mode=log_mode,
            force_vendor=force_vendor,
            handler_options=handler_options,
        )
        _prepare_minimal_sequence_data(sequence_obj, load_mode)
        return _apply_sequence_load_policy(
            sequence_obj,
            load_mode,
            category=sequence_entry["category"],
            reco_options=reco_options,
        )

    # An explicit class intentionally bypasses sequence-registry lookup.
    if sequence_class:
        sequence_obj = sequence_class(handler)
        return _apply_sequence_load_policy(sequence_obj, load_mode)

    experiment = MRIExperiment(
        path,
        handler_load_data,
        log_mode=log_mode,
        force_vendor=force_vendor,
        handler_options=handler_options,
        handler=handler,
    )

    if resolve_sequence:
        logger.warning(
            "No registered sequence for method %r (vendor %r); returning "
            "MRIExperiment. Register the method or pass resolve_sequence=False "
            "to load a generic scan intentionally.",
            handler.type,
            _get_sequence_vendor(handler, force_vendor),
        )
    return _apply_base_load_policy(experiment, load_mode)


def _raise_unrecognized_scan(path: str | Path, force_vendor: str | None) -> None:
    """Raise the loading error for a path that no vendor handler accepted."""
    vendors = ", ".join(sorted(get_vendor_registry().list_vendors()))
    if not Path(path).exists():
        reason = "the path does not exist"
    elif force_vendor:
        reason = (
            f"force_vendor={force_vendor!r} names no registered vendor "
            f"(registered: {vendors})"
        )
    else:
        reason = (
            "no registered vendor recognizes it as a scan "
            f"(registered: {vendors}); pass force_vendor to choose one"
        )
    raise DataLoadingError(
        f"Cannot load scan {path}: {reason}.",
        path=str(path),
        vendor=force_vendor,
    )


def create_experiment(
    path: str | Path,
    load_data: bool | str = False,
    sequence_class: type | None = None,
    force_vendor: str | None = None,
    handler_options: dict[str, Any] | None = None,
    log_mode: str = "error",
) -> MRIExperiment:
    """Compatibility wrapper around :func:`load_scan`.

    Existing callers can continue to use the historical ``load_data``
    argument. New code should prefer :func:`load_scan` and its explicit
    ``mode`` argument.
    """
    return load_scan(
        path,
        mode=load_data,
        sequence_class=sequence_class,
        force_vendor=force_vendor,
        handler_options=handler_options,
        log_mode=log_mode,
    )


def load_external_experiment(
    source: Union[str, Path],
    *,
    matrix: Optional[Sequence[int]] = None,
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    fov_mm: Optional[Sequence[float]] = None,
    sampling_duration_s: Optional[float] = None,
    dwell_time_s: Optional[float] = None,
    bandwidth_hz: Optional[float] = None,
    bandwidth_ppm: Optional[float] = 1,  # default to 1 ppm --- IGNORE ---
    data: Optional[np.ndarray] = None,
    data_domain: str = "unknown",
    data_role: str = "data",
    data_attrs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dtype: Union[str, np.dtype] = np.complex64,
    interleaved: bool = False,
    interleaved_dtype: Union[str, np.dtype] = np.float32,
    data_scaling: float = 1.0,
    sequence_name: str = "external",
    experiment_id: Optional[str] = None,
    load_data: bool = True,
    log_mode: str = "error",
    wrap_sequence: bool = False,
    sequence_class: Optional[Type] = None,
    n_acquisition: Optional[int] = None,
    center_ppm: Optional[float] = 0,
    cut_off: Optional[int] = 0,
) -> MRIExperiment:
    """Create an :class:`MRIExperiment` from externally supplied data arrays.

    Parameters
    ----------
    source
        Path identifying the dataset on disk. Can point to a directory that will
        host supplementary artefacts generated later.
    matrix, dims, coords
        Geometry definition forwarded to :class:`ExternalDataHandler` describing
        the target :class:`xarray.DataArray` layout.
    fov_mm, sampling_duration_s, dwell_time_s, bandwidth_hz
        Acquisition metadata used to populate coordinate attributes.
    data
        Optional ``numpy`` array containing the primary data payload. If
        ``None``, the handler expects to load the data lazily.
    data_domain, data_role, data_attrs
        Metadata that describes what the provided array represents and how it is
        positioned in subsequent pipelines.
    metadata
        Arbitrary experiment metadata stored alongside the array.
    dtype, interleaved, interleaved_dtype
        Data type configuration for the stored payload. ``interleaved`` flags
        interleaved real/imaginary encodings.
    data_scaling
        Multiplicative factor applied to the provided ``data`` before exposure.
    sequence_name, experiment_id
        Logical identifiers used to populate :class:`ExperimentMetadata`.
    load_data
        When ``True``, the external handler converts the provided ``data`` into
        the canonical ``xarray`` representation immediately.
    log_mode
        Logging verbosity hint forwarded to the experiment.
    wrap_sequence
        When ``True``, attempt to wrap the resulting experiment in a registered
        sequence matching ``sequence_name``.
    sequence_class
        Explicit sequence class to use for wrapping. Overrides ``wrap_sequence``
        lookups when provided.

    Returns
    -------
    MRIExperiment
        Experiment instance backed by an ``external`` vendor handler that
        packages the supplied data.

    Raises
    ------
    Exception
        Errors of the sequence constructor selected by ``wrap_sequence`` or
        ``sequence_class`` propagate unchanged; the data are never returned as
        a generic ``MRIExperiment`` in their place.
    """
    registry = get_vendor_registry()

    # Ensure the external vendor handler is registered even if the YAML
    # configuration was loaded before the external handler existed.
    if registry.get_vendor_info("external") is None:
        from hypermri.vendors.external import ExternalDataHandler

        registry.register_vendor(
            "external",
            ExternalDataHandler,
            detection_patterns=[],
            priority=-100,
        )

    normalised_matrix: Optional[Sequence[int]] = (
        tuple(int(dim) for dim in matrix) if matrix is not None else None
    )
    normalised_dims: Optional[Sequence[str]] = (
        tuple(str(dim) for dim in dims) if dims is not None else None
    )

    prepared_data: Optional[np.ndarray] = None
    if data is not None:
        prepared_data = np.asarray(data)

        if normalised_matrix is None:
            normalised_matrix = prepared_data.shape
        else:
            expected_size = int(np.prod(normalised_matrix))
            if prepared_data.size != expected_size:
                raise ValueError(
                    "Provided data does not match the requested matrix size: "
                    f"matrix={normalised_matrix}, data.shape={prepared_data.shape}"
                )
            if prepared_data.shape != tuple(normalised_matrix):
                prepared_data = prepared_data.reshape(normalised_matrix)

    normalised_coords: Optional[Mapping[str, Any]]
    if coords is None:
        normalised_coords = None
    elif isinstance(coords, Mapping):
        normalised_coords = coords
    else:
        if normalised_dims is None:
            raise ValueError(
                "When coords are provided as a sequence, dims must also be supplied."
            )
        coord_sequence = list(coords)
        if len(coord_sequence) != len(normalised_dims):
            raise ValueError(
                "Number of coord entries does not match dims: "
                f"dims={normalised_dims}, coords={len(coord_sequence)}"
            )
        normalised_coords = dict(zip(normalised_dims, coord_sequence, strict=False))

    handler_options: Dict[str, Any] = {
        "matrix": normalised_matrix,
        "dims": normalised_dims,
        "coords": normalised_coords,
        "fov_mm": fov_mm,
        "sampling_duration_s": sampling_duration_s,
        "dwell_time_s": dwell_time_s,
        "bandwidth_hz": bandwidth_hz,
        "data": prepared_data,
        "data_domain": data_domain,
        "data_role": data_role,
        "data_attrs": data_attrs or {},
        "metadata": metadata or {},
        "dtype": dtype,
        "interleaved": interleaved,
        "interleaved_dtype": interleaved_dtype,
        "data_scaling": data_scaling,
        "sequence_name": sequence_name,
        "experiment_id": experiment_id,
        "n_acquisition": n_acquisition,
        "center_ppm": center_ppm,
        "cut_off": cut_off,
        "bandwidth_ppm": bandwidth_ppm,
    }
    if bandwidth_hz is None and dwell_time_s:
        handler_options["bandwidth_hz"] = 1.0 / dwell_time_s
    if dwell_time_s is None and bandwidth_hz:
        handler_options["dwell_time_s"] = 1.0 / bandwidth_hz
    if n_acquisition is None and sampling_duration_s and dwell_time_s:
        handler_options["n_acquisition"] = int(
            round(sampling_duration_s / dwell_time_s)
        )

    handler_options = {k: v for k, v in handler_options.items() if v is not None}

    experiment = MRIExperiment(
        source,
        load_data=load_data,
        log_mode=log_mode,
        force_vendor="external",
        handler_options=handler_options,
    )

    handler = experiment.handler
    primary_payload = None
    primary_role = "data"
    if handler is not None:
        handler_primary = getattr(handler, "primary_data", None)
        if handler_primary is not None:
            handler_role = getattr(handler, "_data_role", None) or "data"
            primary_payload = handler_primary
            primary_role = handler_role
            try:
                experiment.set_primary_data(handler_primary, data_type=handler_role)
            except ValueError:
                if handler_role in {"scanner_data", "seq2d"}:
                    experiment.scanner_data = handler_primary
                else:
                    experiment.data = handler_primary

    if sequence_class:
        sequence_obj = sequence_class(handler)
        if primary_payload is not None:
            try:
                sequence_obj.set_primary_data(primary_payload, data_type=primary_role)
            except ValueError:
                if primary_role in {"scanner_data", "seq2d"}:
                    sequence_obj.scanner_data = primary_payload
                else:
                    sequence_obj.data = primary_payload
        return sequence_obj

    if wrap_sequence:
        smart_class = None
        experiment_type = experiment.type

        if experiment_type and experiment_type.lower() != "unknown":
            smart_class = _resolve_sequence_class(experiment_type, vendor="external")
            if smart_class is None:
                smart_class = _resolve_sequence_class(experiment_type)

        if smart_class:
            sequence_obj = smart_class(handler)
            if primary_payload is not None:
                try:
                    sequence_obj.set_primary_data(
                        primary_payload, data_type=primary_role
                    )
                except ValueError:
                    if primary_role in {"scanner_data", "seq2d"}:
                        sequence_obj.scanner_data = primary_payload
                    else:
                        sequence_obj.data = primary_payload
            return sequence_obj
        else:
            warnings.warn(
                "Requested wrap_sequence for external experiment but no matching "
                "sequence class was found. Returning base MRIExperiment.",
                stacklevel=2,
            )

    return experiment
