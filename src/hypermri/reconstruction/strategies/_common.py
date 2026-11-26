"""Helpers shared by several reconstruction strategies.

Each strategy module imports the names it uses explicitly from here.
"""

from copy import deepcopy

import numpy as np

from hypermri.utils.system.logging import debug_log, init_default_logger
from hypermri.utils.time.temporal import calc_inversion_time_multi_slice

# initialize logger
logger = init_default_logger(__name__)


def _bruker_reco_provenance(handler) -> dict:
    """Snapshot the selected reco file and its shape/timestamp inputs.

    Paths are relative to the scan folder. Values are copied so subsequent
    file selection or dictionary mutation cannot change an existing result.
    No files are read and no numerical default is introduced here.
    """
    source = getattr(handler, "reco_parameter_source", None)
    if source is None:
        return {}
    provenance = {"bruker_reco_parameter_source": source}
    reco = getattr(handler, "reco", None) or {}
    for key, attribute in (
        ("RECO_time", "bruker_reco_time"),
        ("RecoObjectsPerRepetition", "bruker_reco_objects_per_repetition"),
    ):
        if key in reco:
            provenance[attribute] = deepcopy(reco[key])
    return provenance


DEBUG_FILE = "/tmp/recon_strategy_debug.log"
DEBUG_LOG = False


def debug_log_recon_strategy(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


def _create_inversion_idx_coordinate(
    handler, n_inversions_final, n_slices_final, shape_dict, *, inversion_times_ms=None
):
    """Create slice-dependent inversion time coordinate for FAIR_RARE sequences.

    In multi-slice FAIR_RARE acquisitions, different slices are acquired at different
    time points after the inversion pulse, resulting in different effective inversion times
    for each slice. This function calculates the actual inversion time for each
    (slice, inversion_idx) combination.

    Parameters
    ----------
    handler : VendorHandler
        Data handler object containing sequence parameters (slice_order, inversion_time_ms, etc.)
    n_inversions_final : int
        Number of inversion time points in the acquisition
    n_slices_final : int
        Number of slices in the acquisition
    shape_dict : dict
        Dictionary mapping dimension names to their sizes
    inversion_times_ms : array-like, optional
        Explicit base times from the selected scanner reconstruction, in ms.
        Defaults to method times. FAIR slice offsets are applied identically
        to either source; T1 EPI ``IRTime`` is used without FAIR offsets.

    Returns
    -------
    tuple
        Coordinate tuple in format: (("slice_idx", "inversion_idx"), effective_ti_ms, {"unit": "ms"})
        Returns simple 1D coordinate if slice-dependent calculation fails or is not needed
    """
    try:
        # Get base inversion times
        handler.inversion_time_ms
        base_inversion_time_ms = np.atleast_1d(
            handler.inversion_time_ms
            if inversion_times_ms is None
            else inversion_times_ms
        )

        if getattr(handler, "inversion_time_source", None) == "IRTime":
            return (
                "inversion_idx",
                base_inversion_time_ms[:n_inversions_final],
                {"unit": "ms", "source_parameter": "IRTime"},
            )

        # Calculate per-slice time increment
        slice_TI_increment_ms = calc_inversion_time_multi_slice(scan=handler)

        # If increment is zero or negligible, return simple 1D coordinate
        if slice_TI_increment_ms == 0 or abs(slice_TI_increment_ms) < 1e-6:
            logger.debug(
                "Slice TI increment is zero, using simple 1D inversion coordinate"
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        # Get and validate slice order
        slice_order = handler.slice_order
        if slice_order is None or len(slice_order) != n_slices_final:
            logger.warning(
                f"slice_order length ({len(slice_order) if slice_order is not None else 'None'}) "
                f"!= number of slices ({n_slices_final}). Using simple 1D inversion coordinate."
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        if not np.all(np.sort(slice_order) == np.arange(n_slices_final)):
            logger.warning(
                "slice_order does not contain unique indices from 0 to N_slices-1. "
                "Using simple 1D inversion coordinate."
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        # Calculate acquisition rank for each physical slice index
        # Map acquisition order back onto the physical slice index: slice_order[i]
        # stores which physical slice was acquired at position i.
        slice_order = np.asarray(slice_order, dtype=int)
        acquisition_rank = np.empty_like(slice_order)
        acquisition_rank[slice_order] = np.arange(n_slices_final)

        logger.info(
            f"[DIAGNOSTIC] Acquisition rank calculation:\n"
            f"  - slice_order: {slice_order}\n"
            f"  - acquisition_rank: {acquisition_rank}\n"
            f"  - slice_TI_increment: {slice_TI_increment_ms:.2f} ms"
        )

        # Calculate offset for each slice based on its rank
        slice_offsets_ms = acquisition_rank * slice_TI_increment_ms

        # Create the 2D effective TI array using broadcasting
        # Ensure base_inversion_time_ms covers all expected inversions
        if len(base_inversion_time_ms) < n_inversions_final:
            logger.warning(
                f"Base inversion times array length ({len(base_inversion_time_ms)}) "
                f"is less than expected inversions ({n_inversions_final}). Padding with zeros."
            )
            padded_tis = np.pad(
                base_inversion_time_ms,
                (0, n_inversions_final - len(base_inversion_time_ms)),
                mode="constant",
                constant_values=0,
            )
        else:
            padded_tis = base_inversion_time_ms[:n_inversions_final]

        # Broadcast: slice_offsets_ms (n_slices, 1) + padded_tis (1, n_inversions) -> (n_slices, n_inversions)
        effective_ti_ms = slice_offsets_ms[:, np.newaxis] + padded_tis[np.newaxis, :]

        logger.info(
            f"[DIAGNOSTIC] Created slice-dependent inversion coordinate:\n"
            f"  - Coordinate shape: {effective_ti_ms.shape} (slice_idx, inversion_idx)\n"
            f"  - n_slices: {n_slices_final}, n_inversions: {n_inversions_final}\n"
            f"  - Slice TI increment: {slice_TI_increment_ms:.2f} ms\n"
            f"  - Slice order: {slice_order}\n"
            f"  - Base inversion times (first 3): {base_inversion_time_ms[:3]} ms\n"
            f"  - Slice 0 TI values: {effective_ti_ms[0, :3]}... ms\n"
            f"  - Slice 1 TI values: {effective_ti_ms[1, :3] if n_slices_final > 1 else 'N/A'}... ms"
        )

        # Return as multidimensional coordinate
        return (("slice_idx", "inversion_idx"), effective_ti_ms, {"unit": "ms"})

    except (AttributeError, KeyError) as e:
        logger.warning(
            f"Failed to create slice-dependent inversion coordinate: {e}. "
            "Falling back to simple 1D coordinate."
        )
        # Fallback to simple 1D coordinate
        base = np.atleast_1d(getattr(handler, "inversion_time_ms", []))
        ti_coords = (
            base[:n_inversions_final]
            if base.size >= n_inversions_final
            else np.arange(n_inversions_final)
        )
        return ("inversion_idx", ti_coords, {"unit": "ms"})


__all__ = [
    "DEBUG_FILE",
    "debug_log_recon_strategy",
    "logger",
    "_create_inversion_idx_coordinate",
]
